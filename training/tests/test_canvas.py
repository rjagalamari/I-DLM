"""Step-2 packed-canvas builder tests."""

from __future__ import annotations

import torch

from relay_canvas import apply_carry, build_step2_inputs
from relay_verify import block_anchors


def _clean_batch(seq_len=18, block_size=3, prompt=3):
    input_ids = torch.arange(seq_len).unsqueeze(0) + 10
    labels = torch.arange(seq_len).unsqueeze(0) + 10
    labels[:, :prompt] = -100
    return input_ids, labels, block_size


def test_canvas_positions_match_formula():
    # L=18, B=3 → block 2 has s_b=6. Use a_b=[0,0,1,0,0,0] so block 2 matches
    # the relative worked example (accept 1 of 3).
    input_ids, labels, B = _clean_batch()
    a_b = torch.tensor([[0, 0, 1, 0, 0, 0]])
    out = build_step2_inputs(input_ids, labels, a_b, B, mask_token_id=1)
    s_b, o_b = block_anchors(a_b, B)
    assert o_b[0, 2].item() == 6 + 1 + 2 == 9
    # slots 6,7,8 belong to block 2 and draft 9,10,11
    assert out["position_ids_noisy"][0, 6:9].tolist() == [9, 10, 11]
    assert out["anchor"][0, 6:9].tolist() == [9, 9, 9]


def test_worked_relative_all_accept_and_first_reject():
    input_ids, labels, B = _clean_batch()
    a_all = torch.tensor([[3, 3, 3, 3, 3, 3]])
    out = build_step2_inputs(input_ids, labels, a_all, B, mask_token_id=1)
    # block 0: o=0+3+2=5, drafts 5,6,7
    assert out["position_ids_noisy"][0, 0:3].tolist() == [5, 6, 7]
    a0 = torch.zeros(1, 6, dtype=torch.long)
    out0 = build_step2_inputs(input_ids, labels, a0, B, mask_token_id=1)
    # block 0: o=2, drafts 2,3,4
    assert out0["position_ids_noisy"][0, 0:3].tolist() == [2, 3, 4]


def test_clamp_past_sequence_end():
    input_ids, labels, B = _clean_batch(seq_len=9)
    # last block s_b=6, a_b=3 → o_b=11 >= 9 → drop block
    a_b = torch.tensor([[0, 0, 3]])
    out = build_step2_inputs(input_ids, labels, a_b, B, mask_token_id=1)
    assert not out["keep_slot"][0, 6:9].any()
    assert (out["labels2"][0, 6:9] == -100).all()
    assert (out["carry_index_map"][0, 6:9] == -1).all()


def test_carry_index_map_position_aligned():
    input_ids, labels, B = _clean_batch(seq_len=12, prompt=0)
    # block 0 a_b=1 → o=3, drafts 3,4,5. Step-1 block occupied 0,1,2.
    # carry only for drafting pos in [3, 2] → empty. a_b=0 → o=2, drafts 2,3,4
    # carry pos 2 (still in [0,2]).
    a_b = torch.tensor([[0, 0, 0, 0]])
    out = build_step2_inputs(input_ids, labels, a_b, B, mask_token_id=1)
    # slot 0 drafts pos 2 → carry from step-1 slot 2
    assert out["position_ids_noisy"][0, 0].item() == 2
    assert out["carry_index_map"][0, 0].item() == 2
    # slot 1 drafts pos 3, step-1 block 0 only went to 2 → no carry
    assert out["position_ids_noisy"][0, 1].item() == 3
    assert out["carry_index_map"][0, 1].item() == -1


def test_prompt_guard_labels():
    input_ids, labels, B = _clean_batch(seq_len=12, prompt=4)
    a_b = torch.zeros(1, 4, dtype=torch.long)
    out = build_step2_inputs(input_ids, labels, a_b, B, mask_token_id=1)
    # any drafting pos that is prompt (or whose target is prompt) is -100
    for slot in range(12):
        pos = out["position_ids_noisy"][0, slot].item()
        if out["keep_slot"][0, slot] and (labels[0, pos] == -100 or (pos + 1 < 12 and labels[0, pos + 1] == -100)):
            assert out["labels2"][0, slot].item() == -100


def test_apply_carry_and_rollout_only():
    h = torch.randn(1, 6, 4)
    carry = torch.tensor([[2, -1, 0, -1, -1, -1]])
    relay_h, relay_mask = apply_carry(h, carry, rollout_only=False)
    assert relay_mask.tolist() == [[True, False, True, False, False, False]]
    assert torch.allclose(relay_h[0, 0], h[0, 2])
    assert torch.allclose(relay_h[0, 2], h[0, 0])
    _, mask_off = apply_carry(h, carry, rollout_only=True)
    assert not mask_off.any()


def test_noisy_ids_are_masks():
    input_ids, labels, B = _clean_batch()
    a_b = torch.zeros(1, 6, dtype=torch.long)
    out = build_step2_inputs(input_ids, labels, a_b, B, mask_token_id=99)
    assert (out["noisy_ids"] == 99).all()
