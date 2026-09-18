"""Shared fixtures for relay-warmstarted I-DLM tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SFT_DIR = REPO / "llama_factory_sdar" / "src" / "llamafactory" / "train" / "sft"
B7_MODEL = REPO / "model" / "Qwen3-8B-b7-allmasked"

for p in (str(SFT_DIR), str(B7_MODEL)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def tiny_config():
    from configuration_sdar import SDARConfig

    return SDARConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=False,
        use_regular_causal=True,
        fuse_cross_entropy=False,
        attn_implementation="sdpa",
        block_size=3,
        mask_token_id=1,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )


@pytest.fixture
def tiny_model(tiny_config, device):
    from modeling_sdar import SDARForCausalLM

    torch.manual_seed(0)
    model = SDARForCausalLM(tiny_config)
    model.to(device)
    model.train()
    return model
