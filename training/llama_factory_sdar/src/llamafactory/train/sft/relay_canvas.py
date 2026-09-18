"""Step-2 packed-canvas builder for relay-warmstarted I-DLM.

The noisy half stays length ``L``: block ``b`` occupies slots
``[b*B, (b+1)*B)`` and drafts positions ``o_b, o_b+1, ..., o_b+B-1``
where ``o_b = s_b + a_b + 2``. Overlapping canvases share position ids
but never share slots.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:
    from .relay_mask import concat_slot_metadata
    from .relay_verify import block_anchors
except ImportError:  # loaded as a standalone file in unit tests
    from relay_mask import concat_slot_metadata
    from relay_verify import block_anchors


def build_step2_inputs(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    a_b: torch.Tensor,
    block_size: int,
    mask_token_id: int,
    pad_token_id: Optional[int] = None,
) -> dict:
    """Construct pass-2 noisy-half tensors and carry map.

    Args:
        input_ids: ``(B, L)`` clean tokens.
        labels: ``(B, L)`` with ``-100`` on prompt / pad.
        a_b: ``(B, n_blocks)`` accept counts from pass 1.
        block_size: ``B`` (training block / number of masks per canvas).
        mask_token_id: id written into every noisy slot.
        pad_token_id: optional extra exclude.

    Returns:
        dict with
        - ``noisy_ids`` ``(B, L)`` all masks
        - ``position_ids_noisy`` ``(B, L)`` drafting positions
        - ``labels2`` ``(B, L)`` Dream-shifted targets (token at pos+1)
        - ``block_id`` ``(B, L)`` canvas id
        - ``anchor`` ``(B, L)`` per-slot ``o_b``
        - ``carry_index_map`` ``(B, L)`` step-1 noisy slot or ``-1``
        - ``keep_slot`` ``(B, L)`` bool
        - ``s_b``, ``o_b``
        - plus 2L metadata via ``concat_slot_metadata``
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be (B, L), got {input_ids.shape}")
    batch, seq_len = input_ids.shape
    device = input_ids.device
    n_blocks = a_b.shape[-1]
    expected_blocks = (seq_len + block_size - 1) // block_size
    if n_blocks != expected_blocks:
        raise ValueError(
            f"a_b has {n_blocks} blocks, expected {expected_blocks} for L={seq_len}, B={block_size}"
        )

    s_b, o_b = block_anchors(a_b, block_size)
    slots = torch.arange(seq_len, device=device)
    block = (slots // block_size).clamp(max=n_blocks - 1)
    off = slots % block_size
    block_b = block.unsqueeze(0).expand(batch, -1)
    anchor = o_b.gather(1, block_b)
    pos = anchor + off
    s_b_slot = s_b.gather(1, block_b)

    keep_block = o_b < seq_len
    keep_slot = keep_block.gather(1, block_b) & (pos < seq_len)

    # Prompt / pad guard on the drafting position itself and on the target.
    pos_clamped = pos.clamp(max=seq_len - 1)
    target_idx = (pos + 1).clamp(max=seq_len - 1)
    label_at_pos = labels.gather(1, pos_clamped)
    label_at_tgt = labels.gather(1, target_idx)
    prompt_or_pad = label_at_pos.eq(-100) | label_at_tgt.eq(-100)
    if pad_token_id is not None:
        prompt_or_pad = prompt_or_pad | input_ids.gather(1, pos_clamped).eq(pad_token_id)
    past_end = pos + 1 >= seq_len
    supervise = keep_slot & ~prompt_or_pad & ~past_end

    labels2 = torch.full((batch, seq_len), -100, device=device, dtype=labels.dtype)
    labels2 = torch.where(supervise, label_at_tgt, labels2)

    # Warmstart iff the drafting position still sits inside the step-1 block:
    # o_b <= j <= s_b + B - 1. Since j = o_b + off, this is j <= s_b + B - 1.
    carry = keep_slot & (pos <= s_b_slot + block_size - 1) & (pos >= 0)
    carry_index_map = torch.full((batch, seq_len), -1, device=device, dtype=torch.long)
    carry_index_map = torch.where(carry, pos, carry_index_map)

    noisy_ids = torch.full((batch, seq_len), mask_token_id, device=device, dtype=input_ids.dtype)
    position_ids_noisy = torch.where(keep_slot, pos, torch.zeros_like(pos))

    meta = concat_slot_metadata(
        block_b, anchor, position_ids_noisy, seq_len=seq_len, block_size=block_size
    )
    return {
        "noisy_ids": noisy_ids,
        "position_ids_noisy": position_ids_noisy,
        "labels2": labels2,
        "block_id": block_b,
        "anchor": anchor,
        "carry_index_map": carry_index_map,
        "keep_slot": keep_slot,
        "s_b": s_b,
        "o_b": o_b,
        "concat_block_id": meta["block_id"],
        "concat_anchor": meta["anchor"],
        "concat_x0_flag": meta["x0_flag"],
        "concat_position": meta["position"],
    }


def apply_carry(
    step1_h: torch.Tensor,
    carry_index_map: torch.Tensor,
    rollout_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather step-1 hidden states into pass-2 noisy slots.

    ``step1_h`` is ``(B, L, D)`` (noisy-half last-layer states).
    Returns ``relay_h`` ``(B, L, D)`` and ``relay_mask`` ``(B, L)``.
    When ``rollout_only`` is True, the mask is all-False (no injection).
    """
    batch, seq_len, dim = step1_h.shape
    valid = carry_index_map >= 0
    safe = carry_index_map.clamp(min=0)
    # gather along sequence: index (B, L, 1) -> (B, L, D)
    gathered = step1_h.gather(1, safe.unsqueeze(-1).expand(-1, -1, dim))
    relay_h = gathered * valid.unsqueeze(-1).to(gathered.dtype)
    relay_mask = valid if not rollout_only else torch.zeros_like(valid)
    return relay_h, relay_mask


def pack_concat_inputs(
    noisy_ids: torch.Tensor,
    clean_ids: torch.Tensor,
    position_ids_noisy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[noisy | clean]`` concat matching ``prepare_for_bd_training`` layout."""
    seq_len = clean_ids.shape[-1]
    clean_pos = torch.arange(seq_len, device=clean_ids.device).unsqueeze(0).expand_as(clean_ids)
    concat_ids = torch.cat([noisy_ids, clean_ids], dim=-1)
    concat_pos = torch.cat([position_ids_noisy, clean_pos], dim=-1)
    return concat_ids, concat_pos


def build_pass2_attention(step2: dict, use_regular_causal: bool = True):
    """Dense bool ``(B, 2L, 2L)`` on CPU; flex ``BlockMask`` on CUDA if available."""
    try:
        from modeling_sdar import build_attention_from_metadata

        return build_attention_from_metadata(
            step2["concat_block_id"],
            step2["concat_anchor"],
            step2["concat_x0_flag"],
            step2["concat_position"],
            use_regular_causal,
        )
    except ImportError:
        pass
    from relay_mask import materialize_anchor_mask

    bid = step2["concat_block_id"]
    if bid.ndim == 1:
        return materialize_anchor_mask(
            bid, step2["concat_anchor"], step2["concat_x0_flag"], step2["concat_position"], use_regular_causal
        )
    return torch.stack(
        [
            materialize_anchor_mask(
                bid[i],
                step2["concat_anchor"][i],
                step2["concat_x0_flag"][i],
                step2["concat_position"][i],
                use_regular_causal,
            )
            for i in range(bid.size(0))
        ],
        dim=0,
    )


def make_pass2_layout(
    input_ids: torch.Tensor,
    step2: dict,
    use_regular_causal: bool = True,
    attention_mask=None,
) -> dict:
    """Pack ``[noisy|clean]`` tensors the b7 ``forward`` consumes via ``relay_layout``."""
    concat_ids, concat_pos = pack_concat_inputs(
        step2["noisy_ids"], input_ids, step2["position_ids_noisy"]
    )
    keep_half = step2["labels2"] != -100
    keep_full = torch.cat([keep_half, torch.zeros_like(keep_half)], dim=-1)
    if attention_mask is None:
        attention_mask = build_pass2_attention(step2, use_regular_causal)
    return {
        "concat_input_ids": concat_ids,
        "concat_position_ids": concat_pos,
        "attention_mask": attention_mask,
        "logits_to_keep_half": keep_half,
        "logits_to_keep": keep_full,
        "p_mask": None,
        "shifted_labels": step2["labels2"],
    }
