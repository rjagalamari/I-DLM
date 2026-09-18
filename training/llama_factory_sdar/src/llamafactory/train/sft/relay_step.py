"""Reusable two-pass relay step (trainer + tests).

Pass 1 is the standard I-DLM combined loss (mask CE + balanced clean CE).
Pass 2 is mask-CE only on packed step-2 canvases, optionally with detached
hidden-state injection. Each caller scales by 0.5 and backwards separately.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

try:
    from .relay_canvas import apply_carry, build_step2_inputs, make_pass2_layout
    from .relay_verify import verify_hidden_states
except ImportError:
    from relay_canvas import apply_carry, build_step2_inputs, make_pass2_layout
    from relay_verify import verify_hidden_states


IGNORE_INDEX = -100


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def combined_idlm_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    ce_alpha: float = 0.2,
    loss_auto_balance: bool = True,
    position_ids: Optional[torch.Tensor] = None,
    extra_model_kwargs: Optional[dict] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pass-1 / vanilla I-DLM loss: model task CE + Dream-shifted clean CE."""
    kwargs = dict(extra_model_kwargs or {})
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    outputs = model(input_ids=input_ids, labels=labels, **kwargs)
    task_loss = outputs.loss
    hidden = getattr(outputs, "relay_h_last", None)
    if hidden is None:
        if outputs.hidden_states is None:
            raise RuntimeError("model did not return relay_h_last or hidden_states")
        hidden = outputs.hidden_states[-1]
    seq_len = input_ids.size(-1)
    shifted_labels = labels[:, 1 : min(seq_len + 1, labels.shape[1])].contiguous()
    if shifted_labels.shape[1] < seq_len:
        shifted_labels = F.pad(shifted_labels, (0, seq_len - shifted_labels.shape[1]), value=IGNORE_INDEX)
    base = _unwrap(model)
    clean_logits = base.lm_head(hidden[:, seq_len : seq_len + seq_len, :])
    clean_ce_loss = F.cross_entropy(
        clean_logits.view(-1, clean_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )
    if loss_auto_balance:
        scale = task_loss.detach() / (clean_ce_loss.detach() + 1e-8)
        combined = task_loss + scale * clean_ce_loss
    else:
        combined = task_loss + ce_alpha * clean_ce_loss
    extras = {
        "task_loss": task_loss,
        "clean_ce_loss": clean_ce_loss,
        "combined_loss": combined,
        "relay_h_last": hidden,
        "outputs": outputs,
    }
    return combined, extras


def verify_and_build_pass2(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    hidden_2l: torch.Tensor,
    block_size: int,
    mask_token_id: int,
    pad_token_id: Optional[int] = None,
    rollout_only: bool = False,
    use_regular_causal: bool = True,
) -> dict[str, Any]:
    """No-grad verify + packed canvas + detached carry. Safe to call under ``no_grad``."""
    seq_len = input_ids.size(-1)
    noisy_h = hidden_2l[:, :seq_len]
    clean_h = hidden_2l[:, seq_len:]
    base = _unwrap(model)
    stats = verify_hidden_states(
        noisy_h,
        clean_h,
        base.lm_head,
        labels,
        input_ids,
        block_size,
        pad_token_id=pad_token_id,
    )
    step2 = build_step2_inputs(
        input_ids,
        labels,
        stats["a_b"],
        block_size,
        mask_token_id,
        pad_token_id=pad_token_id,
    )
    relay_h, relay_mask = apply_carry(
        noisy_h.detach(),
        step2["carry_index_map"],
        rollout_only=rollout_only,
    )
    layout = make_pass2_layout(input_ids, step2, use_regular_causal=use_regular_causal)
    valid = stats["valid"]
    accepted = stats["accepted"]
    a_b = stats["a_b"]
    n_valid = valid.sum().clamp(min=1)
    metrics = {
        "relay/mean_accept_rate": (accepted & valid).float().sum() / n_valid.float(),
        "relay/mean_a_b": a_b.float().mean(),
        "relay/frac_all_accept": (a_b == block_size).float().mean(),
        "relay/frac_zero_accept": (a_b == 0).float().mean(),
        "relay/pass2_supervised": (step2["labels2"] != IGNORE_INDEX).float().mean(),
    }
    return {
        "stats": stats,
        "step2": step2,
        "layout": layout,
        "relay_h": relay_h,
        "relay_mask": relay_mask,
        "metrics": metrics,
    }


def pass2_mask_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    layout: dict,
    relay_h: torch.Tensor,
    relay_mask: torch.Tensor,
) -> torch.Tensor:
    """Mask-CE only (no clean CE)."""
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        relay_layout=layout,
        relay_h=relay_h,
        relay_mask=relay_mask,
        output_hidden_states=False,
        return_logits=False,
    )
    return outputs.loss


def acceptance_metrics_to_floats(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v.detach().cpu()) for k, v in metrics.items()}
