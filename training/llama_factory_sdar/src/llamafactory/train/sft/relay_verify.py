"""CPU-reference speculative verify + vectorized per-block accept counts.

Used by the relay trainer and by unit tests. The Triton kernel
``fused_spec_verify_from_logits`` is the GPU path; this module is the
source of truth for accept/reject semantics and for ``a_b`` / anchors.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def cpu_spec_verify_from_logits(
    clean_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    spec_vals: torch.Tensor,
    temperature: float = 1.0,
    alpha: float = 1.0,
    rand_accept: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """p/q accept/reject matching ``fused_spec_verify_from_logits``.

    Args:
        clean_logits: ``(N, V)`` verifier (p) logits.
        draft_logits: ``(N, V)`` draft (q) logits.
        spec_vals: ``(N,)`` drafted token ids.
        temperature: softmax temperature.
        alpha: extra scale on q in the ratio ``p / (alpha * q)``.
        rand_accept: optional ``(N,)`` Uniform(0,1) draws. Generated if omitted.

    Returns:
        ``accepted`` bool ``(N,)``, ``rand_accept`` float ``(N,)``.
    """
    inv_temp = 1.0 / temperature if temperature > 0 else 1.0
    p = F.softmax(clean_logits.float() * inv_temp, dim=-1)
    q = F.softmax(draft_logits.float() * inv_temp, dim=-1)
    spec = spec_vals.long().clamp(min=0)
    p_s = p.gather(-1, spec.unsqueeze(-1)).squeeze(-1)
    q_s = q.gather(-1, spec.unsqueeze(-1)).squeeze(-1)
    denom = q_s * alpha
    ratio = torch.where(denom > 0, p_s / denom, torch.zeros_like(p_s))
    if rand_accept is None:
        rand_accept = torch.rand(ratio.shape, device=ratio.device, dtype=torch.float32)
    accepted = (ratio >= 1.0) | (rand_accept < ratio)
    return accepted, rand_accept


def spec_verify_from_logits(
    clean_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    spec_vals: torch.Tensor,
    temperature: float = 1.0,
    alpha: float = 1.0,
    rand_accept: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatch to Triton on CUDA (no custom rand) or the CPU reference.

    When ``rand_accept`` is provided the CPU path is always used so tests
    can pin the RNG. Returns a bool tensor of shape ``(N,)``.
    """
    if (
        rand_accept is None
        and clean_logits.is_cuda
        and draft_logits.is_cuda
    ):
        try:
            try:
                from .fused_verify_kernel import fused_spec_verify_from_logits
            except ImportError:
                from fused_verify_kernel import fused_spec_verify_from_logits

            accepted, _corr = fused_spec_verify_from_logits(
                clean_logits, draft_logits, spec_vals, temperature=temperature, alpha=alpha
            )
            return accepted.bool()
        except Exception:
            pass
    accepted, _ = cpu_spec_verify_from_logits(
        clean_logits, draft_logits, spec_vals, temperature, alpha, rand_accept
    )
    return accepted


def accept_counts_per_block(
    accepted: torch.Tensor,
    valid: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Vectorized ``a_b``: consecutive accepted drafts before the first reject.

    ``accepted`` / ``valid`` are ``(B, L)`` bool. Invalid (prompt / pad)
    positions do not break the prefix and do not increment the count:

        a_b = (cumprod(accepted | ~valid) & valid).sum(within block)

    Returns ``(B, n_blocks)`` int64, with ``n_blocks = ceil(L / block_size)``.
    Trailing slots past ``L`` in the last block are treated as invalid.
    """
    if accepted.ndim != 2 or valid.ndim != 2:
        raise ValueError(f"expected (B, L) tensors, got {accepted.shape} / {valid.shape}")
    batch, seq_len = accepted.shape
    n_blocks = (seq_len + block_size - 1) // block_size
    pad = n_blocks * block_size - seq_len
    if pad:
        accepted = F.pad(accepted, (0, pad), value=False)
        valid = F.pad(valid, (0, pad), value=False)
    # (B, n_blocks, block_size)
    acc = accepted.view(batch, n_blocks, block_size)
    val = valid.view(batch, n_blocks, block_size)
    continue_mask = acc | ~val
    prefix = continue_mask.cumprod(dim=-1).bool()
    counts = (prefix & val).sum(dim=-1)
    return counts.to(torch.int64)


def block_anchors(a_b: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``s_b = b * B``, ``o_b = s_b + a_b + 2``.

    ``a_b`` is ``(..., n_blocks)``. Returns ``(s_b, o_b)`` broadcast to the
    same shape as ``a_b``.
    """
    n_blocks = a_b.shape[-1]
    s_b = torch.arange(n_blocks, device=a_b.device, dtype=a_b.dtype) * block_size
    # broadcast s_b over leading dims
    while s_b.ndim < a_b.ndim:
        s_b = s_b.unsqueeze(0)
    s_b = s_b.expand_as(a_b)
    o_b = s_b + a_b + 2
    return s_b, o_b


def verify_hidden_states(
    noisy_hidden: torch.Tensor,
    clean_hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    block_size: int,
    pad_token_id: Optional[int] = None,
    temperature: float = 1.0,
    alpha: float = 1.0,
) -> dict:
    """Draft from noisy half, verify against clean half, return per-block stats.

    Hidden tensors are ``(B, L, D)`` (the two halves of the 2L layout).
    ``labels`` / ``input_ids`` are length ``L``. A position ``j`` is
    verifiable when ``labels[j+1] != -100`` (and not pad).
    """
    device = noisy_hidden.device
    batch, seq_len, _ = noisy_hidden.shape
    token_valid = labels.ne(-100)
    if pad_token_id is not None:
        token_valid = token_valid & input_ids.ne(pad_token_id)
    valid = torch.zeros_like(token_valid)
    valid[:, :-1] = token_valid[:, 1:]

    drafts = torch.full((batch, seq_len), -100, device=device, dtype=torch.long)
    accepted = torch.zeros((batch, seq_len), device=device, dtype=torch.bool)

    if valid.any():
        lm_dtype = next(lm_head.parameters()).dtype
        noisy_logits = lm_head(noisy_hidden[valid].to(lm_dtype))
        clean_logits = lm_head(clean_hidden[valid].to(lm_dtype))
        selected_drafts = noisy_logits.argmax(dim=-1)
        selected_accepted = spec_verify_from_logits(
            clean_logits, noisy_logits, selected_drafts, temperature=temperature, alpha=alpha
        )
        drafts[valid] = selected_drafts
        accepted[valid] = selected_accepted

    a_b = accept_counts_per_block(accepted, valid, block_size)
    s_b, o_b = block_anchors(a_b, block_size)
    return {
        "drafts": drafts,
        "accepted": accepted,
        "valid": valid,
        "a_b": a_b,
        "s_b": s_b,
        "o_b": o_b,
    }
