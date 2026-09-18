"""Anchor-parameterized I-DLM attention mask.

Slot-indexed (not position-indexed). ``q_idx`` / ``kv_idx`` are indices
into the packed 2L layout. Per-slot tensors carry block id, anchor, x0
flag, and drafting position.

Step-1 (uniform tiling) is the special case ``anchor[noisy slot] = s_b``.
"""

from __future__ import annotations

from typing import Optional

import torch


def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None, use_regular_causal=True):
    """Original I-DLM mask (slot index == position in each half).

    Kept bit-identical to ``modeling_sdar.block_diff_mask`` so tests can
    prove the anchor parameterization is a strict generalization.
    """
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n
    block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)
    if use_regular_causal:
        block_diagonal = (
            (block_q == block_kv) & (x0_flag_q == 0) & (x0_flag_kv == 0) & (q_idx >= kv_idx)
        )
    else:
        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
    if use_regular_causal:
        causal_mask = (q_idx >= kv_idx) & (x0_flag_kv == 1) & (x0_flag_q == 1)
    else:
        causal_mask = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)
    return block_diagonal | offset_block_causal | causal_mask


def anchor_diff_mask(
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    block_id: torch.Tensor,
    anchor: torch.Tensor,
    x0_flag: torch.Tensor,
    position: torch.Tensor,
    use_regular_causal: bool = True,
) -> torch.Tensor:
    """Mask_mod over slot indices with per-slot metadata tensors of length 2L.

    Rules (regular causal / I-DLM):
    - noisy↔noisy: same block id AND ``q_idx >= kv_idx`` (causal in slot order)
    - noisy→clean: clean position ``<`` query's block anchor
    - clean↔clean: ``q_idx >= kv_idx``
    """
    x0_q = x0_flag[q_idx].bool()
    x0_kv = x0_flag[kv_idx].bool()
    block_q = block_id[q_idx]
    block_kv = block_id[kv_idx]
    pos_kv = position[kv_idx]
    anchor_q = anchor[q_idx]

    if use_regular_causal:
        block_diagonal = (~x0_q) & (~x0_kv) & (block_q == block_kv) & (q_idx >= kv_idx)
        causal_mask = x0_q & x0_kv & (q_idx >= kv_idx)
    else:
        block_diagonal = (block_q == block_kv) & (x0_q == x0_kv)
        causal_mask = x0_q & x0_kv & (block_q >= block_kv)
    offset_block_causal = (~x0_q) & x0_kv & (pos_kv < anchor_q)
    return block_diagonal | offset_block_causal | causal_mask


def uniform_slot_metadata(
    seq_len: int,
    block_size: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.long,
) -> dict:
    """Step-1 metadata: noisy slots 0..L-1, clean slots L..2L-1, ``o_b = s_b``."""
    if device is None:
        device = torch.device("cpu")
    slots = torch.arange(2 * seq_len, device=device, dtype=dtype)
    x0_flag = slots >= seq_len
    half_pos = torch.where(x0_flag, slots - seq_len, slots)
    block_id = half_pos // block_size
    anchor = (half_pos // block_size) * block_size
    return {
        "block_id": block_id,
        "anchor": anchor,
        "x0_flag": x0_flag.to(dtype),
        "position": half_pos,
    }


def materialize_anchor_mask(
    block_id: torch.Tensor,
    anchor: torch.Tensor,
    x0_flag: torch.Tensor,
    position: torch.Tensor,
    use_regular_causal: bool = True,
) -> torch.Tensor:
    """Dense ``(2L, 2L)`` bool mask from length-``2L`` metadata tensors."""
    total = block_id.shape[-1]
    q_idx = torch.arange(total, device=block_id.device)[:, None]
    kv_idx = torch.arange(total, device=block_id.device)[None, :]
    return anchor_diff_mask(
        q_idx, kv_idx, block_id, anchor, x0_flag, position, use_regular_causal
    )


def materialize_legacy_mask(
    seq_len: int,
    block_size: int,
    device: Optional[torch.device] = None,
    use_regular_causal: bool = True,
) -> torch.Tensor:
    """Dense ``(2L, 2L)`` materialization of the original ``block_diff_mask``."""
    if device is None:
        device = torch.device("cpu")
    q_idx = torch.arange(2 * seq_len, device=device)[:, None]
    kv_idx = torch.arange(2 * seq_len, device=device)[None, :]
    return block_diff_mask(
        None, None, q_idx, kv_idx, block_size=block_size, n=seq_len, use_regular_causal=use_regular_causal
    )


def concat_slot_metadata(
    noisy_block_id: torch.Tensor,
    noisy_anchor: torch.Tensor,
    noisy_position: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> dict:
    """Build 2L metadata from per-noisy-slot (length L) tensors + a clean half.

    ``noisy_*`` are ``(L,)`` or ``(B, L)``. Clean half uses the original
    tiling (position = 0..L-1, block = pos // B, unused anchor = s_b).
    """
    squeeze = noisy_block_id.ndim == 1
    if squeeze:
        noisy_block_id = noisy_block_id.unsqueeze(0)
        noisy_anchor = noisy_anchor.unsqueeze(0)
        noisy_position = noisy_position.unsqueeze(0)
    batch, length = noisy_block_id.shape
    device = noisy_block_id.device
    dtype = noisy_block_id.dtype
    clean_pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0).expand(batch, -1)
    clean_block = clean_pos // block_size
    clean_anchor = clean_block * block_size
    block_id = torch.cat([noisy_block_id, clean_block], dim=-1)
    anchor = torch.cat([noisy_anchor, clean_anchor], dim=-1)
    position = torch.cat([noisy_position, clean_pos], dim=-1)
    x0_flag = torch.cat(
        [
            torch.zeros(batch, length, device=device, dtype=dtype),
            torch.ones(batch, seq_len, device=device, dtype=dtype),
        ],
        dim=-1,
    )
    if squeeze:
        return {
            "block_id": block_id[0],
            "anchor": anchor[0],
            "x0_flag": x0_flag[0],
            "position": position[0],
        }
    return {"block_id": block_id, "anchor": anchor, "x0_flag": x0_flag, "position": position}
