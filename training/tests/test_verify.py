"""Unit tests for CPU verify + vectorized accept counts."""

from __future__ import annotations

import torch

from relay_verify import (
    accept_counts_per_block,
    block_anchors,
    cpu_spec_verify_from_logits,
)


def test_always_accept_when_p_equals_q():
    torch.manual_seed(0)
    n, v = 8, 16
    logits = torch.randn(n, v)
    spec = logits.argmax(dim=-1)
    # identical p and q → ratio = 1 → always accept regardless of rand
    accepted, _ = cpu_spec_verify_from_logits(logits, logits, spec, rand_accept=torch.ones(n))
    assert accepted.all()


def test_always_reject_when_p_of_spec_is_zero():
    n, v = 6, 10
    draft = torch.zeros(n, v)
    draft[:, 3] = 10.0  # q peaked on token 3
    spec = torch.full((n,), 3, dtype=torch.long)
    clean = torch.full((n, v), -80.0)
    clean[:, 0] = 10.0
    clean[:, 3] = float("-inf")  # p(3) = 0 → ratio = 0 → reject even at rand=0
    accepted, _ = cpu_spec_verify_from_logits(clean, draft, spec, rand_accept=torch.zeros(n))
    assert (~accepted).all()


def test_accept_counts_all_accept():
    accepted = torch.ones(2, 9, dtype=torch.bool)
    valid = torch.ones(2, 9, dtype=torch.bool)
    a_b = accept_counts_per_block(accepted, valid, block_size=3)
    assert a_b.shape == (2, 3)
    assert (a_b == 3).all()


def test_accept_counts_first_reject():
    accepted = torch.tensor([[False, True, True, True, True, True]])
    valid = torch.ones_like(accepted)
    a_b = accept_counts_per_block(accepted, valid, block_size=3)
    assert a_b[0, 0].item() == 0
    assert a_b[0, 1].item() == 3


def test_accept_counts_mid_block():
    # accept, accept, reject → a_b = 2
    accepted = torch.tensor([[True, True, False]])
    valid = torch.ones_like(accepted)
    a_b = accept_counts_per_block(accepted, valid, block_size=3)
    assert a_b[0, 0].item() == 2


def test_accept_counts_multiple_blocks_different():
    accepted = torch.tensor([[True, True, True, True, False, True, False, False, False]])
    valid = torch.ones_like(accepted)
    a_b = accept_counts_per_block(accepted, valid, block_size=3)
    assert a_b.tolist() == [[3, 1, 0]]


def test_prompt_and_pad_excluded():
    # first 3 positions invalid (prompt); remaining all accepted
    accepted = torch.tensor([[False, False, False, True, True, True]])
    valid = torch.tensor([[False, False, False, True, True, True]])
    a_b = accept_counts_per_block(accepted, valid, block_size=3)
    # leading invalid does not break; block 0 has 0 valid → a_b=0
    # block 1 all accepted → 3
    assert a_b.tolist() == [[0, 3]]


def test_invalid_does_not_break_prefix():
    # valid, invalid, valid-accepted → if we skip invalid, a_b should be 2
    accepted = torch.tensor([[True, False, True]])
    valid = torch.tensor([[True, False, True]])
    a_b = accept_counts_per_block(accepted, valid, block_size=3)
    assert a_b[0, 0].item() == 2


def test_vectorized_matches_python_loop():
    torch.manual_seed(1)
    for _ in range(20):
        b, l, bs = 3, 12, 4
        accepted = torch.rand(b, l) > 0.4
        valid = torch.rand(b, l) > 0.2
        got = accept_counts_per_block(accepted, valid, bs)
        ref = _naive_accept_counts(accepted, valid, bs)
        assert torch.equal(got, ref)


def _naive_accept_counts(accepted: torch.Tensor, valid: torch.Tensor, block_size: int) -> torch.Tensor:
    batch, seq_len = accepted.shape
    n_blocks = (seq_len + block_size - 1) // block_size
    out = torch.zeros(batch, n_blocks, dtype=torch.int64)
    for bi in range(batch):
        for blk in range(n_blocks):
            start = blk * block_size
            end = min(start + block_size, seq_len)
            count = 0
            for pos in range(start, end):
                if not valid[bi, pos]:
                    continue
                if accepted[bi, pos]:
                    count += 1
                else:
                    break
            out[bi, blk] = count
    return out


def test_block_anchors_formula():
    a_b = torch.tensor([[0, 1, 3]])
    s_b, o_b = block_anchors(a_b, block_size=3)
    assert s_b.tolist() == [[0, 3, 6]]
    assert o_b.tolist() == [[2, 6, 11]]  # s + a + 2
