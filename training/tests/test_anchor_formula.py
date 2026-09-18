"""Worked example from the handoff plan, Section 4.3."""

from __future__ import annotations


def _canvas(s_b: int, a_b: int, block_size: int) -> tuple[int, list[int]]:
    o_b = s_b + a_b + 2
    return o_b, list(range(o_b, o_b + block_size))


def test_worked_example_reject_mid():
    # B=3, s_b=8, masks 8,9,10 proposing x_9,x_10,x_11
    # accept x_9, reject x_10 → a_b=1 → new masks at 11,12,13
    o_b, positions = _canvas(s_b=8, a_b=1, block_size=3)
    assert o_b == 11
    assert positions == [11, 12, 13]


def test_worked_example_all_accept():
    # a_b=3 → new masks at 13,14,15
    o_b, positions = _canvas(s_b=8, a_b=3, block_size=3)
    assert o_b == 13
    assert positions == [13, 14, 15]


def test_worked_example_first_reject():
    # a_b=0 → new masks at 10,11,12
    o_b, positions = _canvas(s_b=8, a_b=0, block_size=3)
    assert o_b == 10
    assert positions == [10, 11, 12]


def test_anchors_never_regress():
    s_b = 8
    for a_b in range(0, 4):
        o_b, _ = _canvas(s_b, a_b, 3)
        assert o_b > s_b
