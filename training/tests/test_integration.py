"""Tiny-model two-pass smoke + vanilla-path equivalence."""

from __future__ import annotations

import copy

import pytest
import torch

from relay_step import (
    acceptance_metrics_to_floats,
    combined_idlm_loss,
    pass2_mask_loss,
    verify_and_build_pass2,
)


def _synthetic_batch(device, batch=2, seq_len=12, prompt=3, vocab=64, seed=0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    input_ids = torch.randint(4, vocab, (batch, seq_len), generator=g)
    labels = input_ids.clone()
    labels[:, :prompt] = -100
    return input_ids.to(device), labels.to(device)


def _two_pass_step(model, input_ids, labels, rollout_only: bool):
    combined, extras = combined_idlm_loss(model, input_ids, labels, loss_auto_balance=True)
    hidden = extras["relay_h_last"]
    (combined * 0.5).backward()
    with torch.no_grad():
        built = verify_and_build_pass2(
            model,
            input_ids,
            labels,
            hidden.detach(),
            block_size=model.config.block_size,
            mask_token_id=model.config.mask_token_id,
            pad_token_id=model.config.pad_token_id,
            rollout_only=rollout_only,
        )
    loss2 = pass2_mask_loss(
        model, input_ids, labels, built["layout"], built["relay_h"], built["relay_mask"]
    )
    (loss2 * 0.5).backward()
    total = 0.5 * (combined.detach() + loss2.detach())
    metrics = acceptance_metrics_to_floats(built["metrics"])
    metrics["train/task_loss"] = float(extras["task_loss"].detach())
    metrics["train/combined_loss"] = float(combined.detach())
    metrics["train/pass2_loss"] = float(loss2.detach())
    metrics["train/two_pass_loss"] = float(total)
    return total, metrics


def _run_smoke(tiny_config, device, rollout_only: bool, steps: int = 50):
    from modeling_sdar import SDARForCausalLM

    torch.manual_seed(0)
    model = SDARForCausalLM(tiny_config).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    input_ids, labels = _synthetic_batch(device, seed=1)
    losses = []
    last_metrics = None
    for _ in range(steps):
        opt.zero_grad()
        total, metrics = _two_pass_step(model, input_ids, labels, rollout_only)
        opt.step()
        assert torch.isfinite(total)
        losses.append(float(total))
        last_metrics = metrics
    return losses, last_metrics


def test_tiny_relay_smoke_loss_decreases(tiny_config, device):
    losses, metrics = _run_smoke(tiny_config, device, rollout_only=False, steps=50)
    assert all(map(lambda x: x == x and x != float("inf"), losses))
    # Compare early vs late window to tolerate step noise.
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10
    for key in ("relay/mean_accept_rate", "relay/frac_all_accept", "relay/mean_a_b"):
        assert key in metrics
        assert 0.0 <= metrics["relay/mean_accept_rate"] <= 1.0
        assert 0.0 <= metrics["relay/frac_all_accept"] <= 1.0
    assert 0.0 <= metrics["relay/mean_a_b"] <= tiny_config.block_size


def test_tiny_rollout_only_smoke(tiny_config, device):
    losses, metrics = _run_smoke(tiny_config, device, rollout_only=True, steps=50)
    assert all(map(lambda x: x == x and x != float("inf"), losses))
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10
    assert 0.0 <= metrics["relay/mean_accept_rate"] <= 1.0


def test_vanilla_loss_matches_hidden_states_path(tiny_model, device):
    """relay_h_last path equals the original output_hidden_states[-1] loss."""
    input_ids, labels = _synthetic_batch(device, seed=2)
    torch.manual_seed(11)
    loss_a, extra_a = combined_idlm_loss(
        tiny_model, input_ids, labels, extra_model_kwargs={"output_hidden_states": True}
    )
    # Same weights, same RNG for the (deterministic given p=1) mask path.
    torch.manual_seed(11)
    loss_b, extra_b = combined_idlm_loss(tiny_model, input_ids, labels)
    assert torch.allclose(loss_a, loss_b, atol=1e-6)
    assert torch.allclose(extra_a["relay_h_last"], extra_b["relay_h_last"], atol=1e-6)
    assert extra_a["outputs"].hidden_states is not None


def test_vanilla_does_not_need_relay_layout(tiny_model, device):
    """A vanilla forward never consults relay_layout / injection."""
    input_ids, labels = _synthetic_batch(device, seed=5)
    torch.manual_seed(0)
    out1 = tiny_model(input_ids=input_ids, labels=labels)
    torch.manual_seed(0)
    out2 = tiny_model(input_ids=input_ids, labels=labels, relay_h=None, relay_mask=None, relay_layout=None)
    assert torch.allclose(out1.loss, out2.loss, atol=1e-6)


def test_clone_weights_vanilla_loss_deterministic(tiny_config, device):
    """Same init + same batch + same seed → identical vanilla loss (additive guard)."""
    from modeling_sdar import SDARForCausalLM

    input_ids, labels = _synthetic_batch(device, seed=7)
    torch.manual_seed(99)
    a = SDARForCausalLM(tiny_config).to(device).train()
    b = copy.deepcopy(a)
    torch.manual_seed(123)
    la, _ = combined_idlm_loss(a, input_ids, labels)
    torch.manual_seed(123)
    lb, _ = combined_idlm_loss(b, input_ids, labels)
    assert torch.allclose(la, lb, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU flex/Triton path; login node is CPU-only")
def test_gpu_short_relay_run(tiny_config):
    device = torch.device("cuda")
    losses, metrics = _run_smoke(tiny_config, device, rollout_only=False, steps=5)
    assert torch.isfinite(torch.tensor(losses)).all()
    assert 0.0 <= metrics["relay/mean_accept_rate"] <= 1.0
