"""Relay injection: identity at zero-init, gating, and gradient flow."""

from __future__ import annotations

import torch

from relay_canvas import apply_carry, build_step2_inputs, make_pass2_layout
from relay_step import pass2_mask_loss


def _batch(device, seq_len=12, batch=2, prompt=3, vocab=64):
    torch.manual_seed(3)
    input_ids = torch.randint(4, vocab, (batch, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, :prompt] = -100
    return input_ids, labels


def test_zero_init_identity(tiny_model, device):
    model = tiny_model
    assert torch.count_nonzero(model.relay_layer_norm.weight) == 0
    assert torch.count_nonzero(model.relay_layer_norm.bias) == 0
    input_ids, labels = _batch(device)
    h = torch.randn(input_ids.size(0), input_ids.size(1), model.config.hidden_size, device=device)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask[:, 5:8] = True
    model.eval()
    model.train()
    out_off = model(input_ids=input_ids, labels=labels, return_logits=True)
    out_on = model(
        input_ids=input_ids,
        labels=labels,
        relay_h=h,
        relay_mask=mask,
        return_logits=True,
    )
    assert torch.allclose(out_off.logits, out_on.logits, atol=1e-5, rtol=1e-5)


def test_inject_never_touches_clean_half(tiny_model, device):
    model = tiny_model
    input_ids, labels = _batch(device, batch=1)
    seq_len = input_ids.size(1)
    embeds = model.model.embed_tokens(torch.cat([input_ids, input_ids], dim=-1))
    h = torch.randn(1, seq_len, model.config.hidden_size, device=device)
    mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    # Non-zero LN so a leak would be visible.
    with torch.no_grad():
        model.relay_layer_norm.weight.fill_(1.0)
    injected = model._inject_relay(embeds.clone(), h, mask)
    assert torch.equal(injected[:, seq_len:], embeds[:, seq_len:])
    assert not torch.equal(injected[:, :seq_len], embeds[:, :seq_len])


def test_gamma_ones_changes_only_attending_slots(tiny_model, device):
    """Clean-half logits stay identical; isolated canvases without carry stay identical."""
    model = tiny_model
    torch.manual_seed(4)
    seq_len, block_size = 12, 3
    input_ids = torch.arange(4, 4 + seq_len, device=device).unsqueeze(0)
    labels = input_ids.clone()
    # a_b = 0 on block 0 (has carry at first slot) and all-accept on last block
    # so later canvases start far enough that they may not share carry.
    a_b = torch.tensor([[0, 3, 3, 0]], device=device)
    step2 = build_step2_inputs(input_ids, labels, a_b, block_size, mask_token_id=1)
    relay_h, relay_mask = apply_carry(
        torch.randn(1, seq_len, model.config.hidden_size, device=device),
        step2["carry_index_map"],
        rollout_only=False,
    )
    from relay_canvas import make_pass2_layout

    layout = make_pass2_layout(input_ids, step2)
    with torch.no_grad():
        model.relay_layer_norm.weight.zero_()
        model.relay_layer_norm.bias.zero_()
    out_zero = model(
        input_ids=input_ids,
        labels=labels,
        relay_layout=layout,
        relay_h=relay_h,
        relay_mask=relay_mask,
        return_logits=True,
    )
    out_none = model(
        input_ids=input_ids,
        labels=labels,
        relay_layout=layout,
        return_logits=True,
    )
    assert torch.allclose(out_zero.logits, out_none.logits, atol=1e-5, rtol=1e-5)

    with torch.no_grad():
        model.relay_layer_norm.weight.fill_(1.0)
    out_ones = model(
        input_ids=input_ids,
        labels=labels,
        relay_layout=layout,
        relay_h=relay_h,
        relay_mask=relay_mask,
        return_logits=True,
    )
    logits_z = out_zero.logits
    logits_o = out_ones.logits
    # clean half (slots L:2L) never sees noisy injection
    assert torch.allclose(logits_z[:, seq_len:], logits_o[:, seq_len:], atol=1e-5, rtol=1e-5)
    # at least one noisy slot that received carry must change
    carry_slots = relay_mask[0].nonzero(as_tuple=False).flatten()
    assert carry_slots.numel() > 0
    diffs = (logits_z[0] - logits_o[0]).abs().sum(dim=-1)
    assert diffs[carry_slots].max() > 1e-5


def test_gradients_reach_ln_in_arm_c_not_arm_b(tiny_model, device):
    model = tiny_model
    input_ids, labels = _batch(device, batch=1, seq_len=12, prompt=0)
    block_size = model.config.block_size
    n_blocks = input_ids.size(1) // block_size
    # Force a_b=0 so each canvas has a position-aligned carry slot.
    a_b = torch.zeros(1, n_blocks, dtype=torch.long, device=device)
    step2 = build_step2_inputs(input_ids, labels, a_b, block_size, model.config.mask_token_id)
    hidden = torch.randn(1, input_ids.size(1), model.config.hidden_size, device=device)
    layout = make_pass2_layout(input_ids, step2)
    relay_h, relay_mask = apply_carry(hidden, step2["carry_index_map"], rollout_only=False)
    assert relay_mask.any()
    model.zero_grad()
    loss_c = pass2_mask_loss(model, input_ids, labels, layout, relay_h, relay_mask)
    loss_c.backward()
    grad_c = model.relay_layer_norm.weight.grad
    assert grad_c is not None
    assert torch.isfinite(grad_c).all()
    assert grad_c.abs().sum() > 0

    model.zero_grad()
    relay_h_b, relay_mask_b = apply_carry(hidden, step2["carry_index_map"], rollout_only=True)
    assert not relay_mask_b.any()
    loss_b = pass2_mask_loss(model, input_ids, labels, layout, relay_h_b, relay_mask_b)
    loss_b.backward()
    grad_b = model.relay_layer_norm.weight.grad
    assert grad_b is None or torch.count_nonzero(grad_b) == 0
