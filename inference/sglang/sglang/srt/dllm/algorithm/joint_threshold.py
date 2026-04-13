import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


def _get_num_transfer_tokens(block_size: int, steps: int):
    """Distribute block_size tokens evenly across steps (matches official HF generate)."""
    base = block_size // steps
    remainder = block_size % steps
    schedule = torch.zeros(steps, dtype=torch.int64)
    schedule[:] = base
    schedule[:remainder] += 1
    return schedule


class JointThreshold(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.5)
        self.edit_threshold = config.algorithm_config.get("edit_threshold", 0)
        self.max_post_edit_steps = config.algorithm_config.get(
            "max_post_edit_steps", 16
        )
        self.penalty_lambda = config.algorithm_config.get("penalty_lambda", 0)
        # steps: number of denoising steps per block.
        # Default = block_size → 1 token per step (most conservative, matches HF default).
        # Smaller steps = faster but potentially lower quality.
        self.steps = config.algorithm_config.get("steps", self.block_size)
        # Pre-compute per-step transfer schedule
        self._transfer_schedule = _get_num_transfer_tokens(self.block_size, self.steps)

    def cleanup_request(self, req_pool_idx: int):
        pass

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        overlap_fn=None,
    ) -> tuple[LogitsProcessorOutput | torch.Tensor, torch.Tensor | None, bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device

        mask_index = forward_batch.input_ids == self.mask_id
        if not mask_index.any():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        start_list = []
        prompt_masks = []
        for i in range(batch_size):
            block_start = i * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]

            prompt_mask = block_input_ids != self.mask_id
            prompt_masks.append(prompt_mask)
            start_list.append(prompt_mask.sum().item())

        post_edit_steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Track denoising step index per batch item (for scheduled transfer)
        denoise_step = torch.zeros(batch_size, dtype=torch.int32, device=device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        any_changed_in_last_step = False

        schedule = self._transfer_schedule.to(device)
        max_iterations = self.steps + self.max_post_edit_steps

        for _ in range(max_iterations):
            if finished.all():
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            any_changed_in_last_step = False

            for i in range(batch_size):
                if finished[i]:
                    continue

                block_start = i * self.block_size
                block_end = block_start + self.block_size

                curr_input_ids = forward_batch.input_ids[block_start:block_end]
                curr_logits = logits_output.full_logits[block_start:block_end]
                curr_prompt_mask = prompt_masks[i]

                if self.penalty_lambda > 0:
                    prev_ids = curr_input_ids[:-1]
                    curr_logits[1:, :].scatter_(
                        1, prev_ids.unsqueeze(-1), -self.penalty_lambda, reduce="add"
                    )

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )

                curr_mask_index = curr_input_ids == self.mask_id
                has_mask = curr_mask_index.any()

                # Mask to token (M2T) with scheduled transfer
                mask_transfer_index = torch.zeros_like(curr_mask_index)
                if has_mask:
                    step_idx = min(int(denoise_step[i].item()), self.steps - 1)
                    num_to_transfer = int(schedule[step_idx].item())
                    num_available = int(curr_mask_index.sum().item())
                    num_to_transfer = min(num_to_transfer, num_available)

                    confidence = torch.where(curr_mask_index, p, torch.tensor(-float('inf'), device=device))
                    high_conf = (confidence > self.threshold) & curr_mask_index
                    num_high = int(high_conf.sum().item())

                    if num_high >= num_to_transfer:
                        # Enough high-confidence tokens: commit all of them
                        mask_transfer_index = high_conf
                    else:
                        # Not enough: take top-k by confidence
                        _, idx = torch.topk(confidence, k=num_to_transfer)
                        mask_transfer_index[idx] = True

                    denoise_step[i] += 1
                else:
                    post_edit_steps[i] += 1
                    if post_edit_steps[i] > self.max_post_edit_steps:
                        finished[i] = True
                        continue

                # Token to token (T2T) editing
                edit_mask = ~curr_mask_index & ~curr_prompt_mask
                edit_transfer_index = (
                    (p > self.edit_threshold) & (curr_input_ids != x) & edit_mask
                )

                transfer_index = mask_transfer_index | edit_transfer_index
                if not transfer_index.any():
                    finished[i] = True
                    continue

                curr_input_ids[transfer_index] = x[transfer_index]
                any_changed_in_last_step = True

        if any_changed_in_last_step:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        next_token_ids_list = []
        for i in range(batch_size):
            block_start = i * self.block_size
            block_end = block_start + self.block_size
            block_ids = forward_batch.input_ids[block_start:block_end]
            next_token_ids_list.append(block_ids[start_list[i]:])

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = JointThreshold
