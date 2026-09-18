"""Mask equivalence (uniform anchors) and ragged-canvas isolation."""

from __future__ import annotations

import torch

from relay_mask import (
    concat_slot_metadata,
    materialize_anchor_mask,
    materialize_legacy_mask,
    uniform_slot_metadata,
)


def test_uniform_anchors_match_legacy_mask():
    seq_len, block_size = 12, 3
    legacy = materialize_legacy_mask(seq_len, block_size)
    meta = uniform_slot_metadata(seq_len, block_size)
    modern = materialize_anchor_mask(
        meta["block_id"], meta["anchor"], meta["x0_flag"], meta["position"]
    )
    assert legacy.shape == (24, 24)
    assert torch.equal(legacy, modern)


def test_uniform_anchors_small_n2():
    # paper figure uses N=2 → B=1? Training block_size=2 is also common.
    for seq_len, block_size in [(6, 2), (8, 4), (9, 3)]:
        legacy = materialize_legacy_mask(seq_len, block_size)
        meta = uniform_slot_metadata(seq_len, block_size)
        modern = materialize_anchor_mask(
            meta["block_id"], meta["anchor"], meta["x0_flag"], meta["position"]
        )
        assert torch.equal(legacy, modern), (seq_len, block_size)


def test_ragged_mask_isolation_and_anchors():
    """Two noisy canvases with unequal strides; overlapping positions must not attend.

    L=12, B=4, three blocks. Accepts a = [3, 1, 0] →
    o = [5, 7, 10]. Canvases draft:
      b0: 5,6,7,8
      b1: 7,8,9,10
      b2: 10,11,12,13 (12,13 clamped conceptually; still slots)
    b0 and b1 both draft positions 7 and 8.
    """
    seq_len, block_size = 12, 4
    a_b = torch.tensor([3, 1, 0])
    s_b = torch.arange(3) * block_size
    o_b = s_b + a_b + 2
    slots = torch.arange(seq_len)
    block = slots // block_size
    off = slots % block_size
    noisy_block = block
    noisy_anchor = o_b[block]
    noisy_pos = noisy_anchor + off
    meta = concat_slot_metadata(noisy_block, noisy_anchor, noisy_pos, seq_len, block_size)
    mask = materialize_anchor_mask(
        meta["block_id"], meta["anchor"], meta["x0_flag"], meta["position"]
    )
    # noisy slots 0..11, clean 12..23
    # within canvas 0 (slots 0-3): causal
    assert mask[1, 0]  # slot 1 sees slot 0
    assert not mask[0, 1]  # not the other way
    # cross-canvas isolation: canvas 0 slot drafting pos 7 (slot 2: 5+2)
    # canvas 1 slot drafting pos 7 (slot 4: 7+0)
    assert noisy_pos[2].item() == 7
    assert noisy_pos[4].item() == 7
    assert not mask[2, 4]
    assert not mask[4, 2]
    # clean visibility strictly below each anchor
    # canvas 1 (slots 4-7) has o_b=7: may see clean pos < 7, not pos >= 7
    clean_pos6 = 12 + 6  # clean slot for position 6
    clean_pos7 = 12 + 7
    assert mask[4, clean_pos6]
    assert not mask[4, clean_pos7]
    # clean-half causality unchanged
    assert mask[12 + 5, 12 + 4]
    assert not mask[12 + 4, 12 + 5]
    # noisy cannot see later clean
    assert not mask[0, 12 + 11]
