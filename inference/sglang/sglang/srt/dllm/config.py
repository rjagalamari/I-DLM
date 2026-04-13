from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        mask_id: int,
        max_running_requests: int,
        causal_prefill: bool = False,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        # If True, STAGING_PREFILL uses causal (lower-triangular) attention
        # instead of bidirectional. Required for SDAR models trained with
        # use_regular_causal=True where prompt tokens attend causally.
        self.causal_prefill = causal_prefill

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):
        if server_args.dllm_algorithm is None:
            return None

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )
        DLLM_PARAMS = {
            "LLaDA2MoeModelLM": {"block_size": 32, "mask_id": 156895},
            "SDARForCausalLM": {"block_size": 4, "mask_id": 151669},
            "SDARMoeForCausalLM": {"block_size": 4, "mask_id": 151669},
            "Qwen3ForCausalLM": {"block_size": 4, "mask_id": 151669},
        }

        arch = model_config.hf_config.architectures[0]
        if arch in DLLM_PARAMS:
            params = DLLM_PARAMS[arch]
            block_size = params["block_size"]
            mask_id = params["mask_id"]
        else:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")

        # Prefer block_size from model config if available (e.g. block_size=1
        # models vs the default block_size=4 in DLLM_PARAMS).
        hf_block_size = getattr(model_config.hf_config, "block_size", None)
        if hf_block_size is not None:
            block_size = hf_block_size

        # SDAR models with use_regular_causal=True were trained with causal
        # attention for prompt (x0) tokens; prefill must use causal attention.
        causal_prefill = getattr(model_config.hf_config, "use_regular_causal", False)

        max_running_requests = (
            1
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )

        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(server_args.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f)

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)

        # Allow YAML to override causal_prefill (some models have
        # use_regular_causal=None but still need causal prefill).
        if algorithm_config.get("causal_prefill") is not None:
            causal_prefill = algorithm_config["causal_prefill"]

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            causal_prefill=causal_prefill,
        )
