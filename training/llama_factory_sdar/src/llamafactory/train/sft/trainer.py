# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sys
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import torch.nn.functional as F
import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    ## verifying the noisy logits...
   
    def _relay_verify(self, model, inputs):
        """Verify pass-1 drafts against the clean half.

        Ported from the I-DLM serving path
        (inference/sglang/sglang/srt/dllm/algorithm/fused_verify_kernel.py):
          draft token = argmax(q)        -- "argmax SPEC positions directly"
          p, q        = full-vocab softmax; the verifier applies NO top_k/top_p
          temperature = 1.0 -> INV_TEMP = 1.0, so no logit scaling
          ratio       = p / (q * ALPHA), ALPHA = verify_alpha, default 1.0
          accepted    = (ratio >= 1.0) | (rand < ratio)

        KNOWN APPROXIMATION -- structural, not fixable in this training format:
          At inference, p for spec_k is conditioned on the DRAFT prefix. Here the
          clean half holds ground truth, so p is conditioned on the TRUE prefix.
          This accept rate is a related but different quantity from live acceptance.
        """
        h1 = self._relay["h1"]                          # [B, 2L, H]
        labels = inputs["labels"]
        B, L = labels.shape
        assert B == 1, "answer-span slicing assumes batch size 1"
        base = getattr(model, "module", model)
        lm_dtype = base.lm_head.weight.dtype             # h1 is fp32, lm_head is bf16
        alpha = 1.0                                      # verify_alpha default

        with torch.no_grad():
            m = (labels[0] != -100).nonzero().flatten()
            if m.numel() == 0:                           # truncation removed the answer
                empty_b = torch.zeros(0, dtype=torch.bool, device=h1.device)
                empty_l = torch.zeros(0, dtype=torch.long, device=h1.device)
                return empty_b, empty_l, 0.0, 0, 0

            a, z = int(m[0]), int(m[-1])                 # answer spans a .. z
            assert m.numel() == z - a + 1, "answer span not contiguous"
            assert a >= 1, "answer starts at position 0; nothing can propose it"

            # hidden[i] predicts token i+1, so positions a-1 .. z-1 propose the
            # answer tokens a .. z -- every answer token exactly once.
            h_n = h1[0, a - 1 : z].to(lm_dtype)                  # noisy half -> q
            h_c = h1[0, L + a - 1 : L + z].to(lm_dtype)          # clean half -> p
            n = z - a + 1

            accepted = torch.empty(n, dtype=torch.bool, device=h1.device)
            drafts = torch.empty(n, dtype=torch.long, device=h1.device)
            accepted_ratio = 0.0

            for s in range(0, n, 512):
                e = min(s + 512, n)
                q_logits = base.lm_head(h_n[s:e]).float()
                p_logits = base.lm_head(h_c[s:e]).float()

                draft = q_logits.argmax(-1)
                drafts[s:e] = draft
                q = q_logits.softmax(-1).gather(-1, draft[:, None]).squeeze(-1)
                p = p_logits.softmax(-1).gather(-1, draft[:, None]).squeeze(-1)

                ratio = torch.where(q > 0, p / (q * alpha), torch.zeros_like(p))
                accepted_ratio += ratio.clamp(max=1.0).sum().item()
                accepted[s:e] = (ratio >= 1.0) | (torch.rand_like(ratio) < ratio)

        return accepted, drafts, accepted_ratio / max(n, 1), a, z



    def _relay_scan(self, accepted, a, block_size):
        """Left-to-right prefix scan: within each block, keep drafts until the
        first rejection, then stop.

        UNRESOLVED -- the grouping. block_diff_mask tiles blocks as
        q_idx // block_size counting from position 0, so blocks do not align with
        where the answer starts. This uses the MODEL's tiling, grouping each
        decision by the block of the mask that produced it. The alternative is
        answer-aligned blocks (starting at `a`), which is what inference does.
        Not settled.
        """
        n = accepted.numel()
        keep = torch.zeros_like(accepted)
        alive = {}                                  # block index -> still accepting?
        for k in range(n):
            b = (a - 1 + k) // block_size           # block of the deciding position
            if alive.get(b, True):
                if accepted[k]:
                    keep[k] = True
                else:
                    alive[b] = False
        return keep





    # Relay trainer changes..
    @override
    def training_step(self, model, inputs, *args, **kwargs):
        """Two-pass hidden-state relay rollout.

        Each super() call is a complete forward+backward, so gradients from both
        passes accumulate into .grad before the optimizer step. The constant factor
        of 2 versus the mean is absorbed by Adam.

        Pass 1's backward frees its graph before pass 2 runs, so the relay carry
        must be detached -- this is the Relay (sg) variant.
        """
        self._relay = {"pass": 1}
        loss1 = super().training_step(model, inputs, *args, **kwargs)

        # ---- rollout: verify pass-1 drafts, then apply the per-block prefix rule ----
        accepted, drafts, mean_ratio, a, z = self._relay_verify(model, inputs)
        if accepted.numel() > 0:
            base = getattr(model, "module", model)
            keep = self._relay_scan(accepted, a, base.config.block_size)
            self._relay.update({"drafts": drafts, "keep": keep, "a": a, "z": z})
            self.log({
                "train/accept_rate": accepted.float().mean().item(),
                "train/expected_accept": mean_ratio,
                "train/keep_rate": keep.float().mean().item(),
            })

        self._relay["pass"] = 2
        loss2 = super().training_step(model, inputs, *args, **kwargs)

        self._relay = None          # so eval / prediction_step take the normal path
        return (loss1 + loss2) / 2  # logging only -- both backwards already ran



    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        # NOTE: The "ar" loss assumes an SDAR model that internally concatenates [noisy|clean],
        # producing hidden_states of length 2*seq_len. This will not work with standard AR models.
        if self.finetuning_args.idlm_loss_type != "ar":
            return super().compute_loss(model, inputs, *args, **kwargs)

        alpha = self.finetuning_args.ce_alpha
        labels = inputs.get("labels")
        assert labels is not None

        # Forward pass — the SDAR model internally concatenates [noisy_xt | clean_x0],
        # so input_ids has length L but hidden_states has length 2L.
        outputs = model(**inputs, output_hidden_states=True)
        task_loss = outputs["loss"]  # CE on all positions (noisy + clean)

        # relay pass1 
        # ===== RELAY: keep pass 1's hidden states for pass 2 =====
        if getattr(self, "_relay", None) is not None and self._relay["pass"] == 1:
            self._relay["h1"] = outputs["hidden_states"][-1].detach()   # [B, 2L, H]


        # Unwrap DeepSpeedEngine/FSDP to access lm_head directly
        unwrapped_model = getattr(model, "module", model)

        # seq_len = L (original input length); clean region is hidden_states[:, L:2L]
        seq_len = inputs["input_ids"].size(-1)

        # AR CE loss on clean (x0) region with Dream-shift-aligned labels.
        # hidden[i] predicts token[i+1], so shift labels by 1.
        shifted_labels = labels[:, 1 : min(seq_len + 1, labels.shape[1])].contiguous()
        if shifted_labels.shape[1] < seq_len:
            shifted_labels = F.pad(shifted_labels, (0, seq_len - shifted_labels.shape[1]), value=-100)

        clean_logits = unwrapped_model.lm_head(
            outputs["hidden_states"][-1][:, seq_len : seq_len + seq_len, :].to(
                unwrapped_model.lm_head.weight.dtype
            )
        )
        ce_logits = clean_logits.view(-1, clean_logits.size(-1))
        ce_labels = shifted_labels.view(-1)
        clean_ce_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(ce_logits, ce_labels)

        if self.finetuning_args.loss_auto_balance:
            scale = task_loss.detach() / (clean_ce_loss.detach() + 1e-8)
            combined_loss = task_loss + scale * clean_ce_loss
        else:
            combined_loss = task_loss + alpha * clean_ce_loss

        # Logging
        log_dict = {
            "train/task_loss": task_loss.item(),
            "train/clean_ce_loss": clean_ce_loss.item(),
            "train/combined_loss": combined_loss.item(),
            "train/alpha": alpha,
        }
        self.log(log_dict)

        if self.state.global_step % self.args.logging_steps == 0:
            logger.info_rank0(
                f"Step {self.state.global_step}: "
                f"task_loss={task_loss.item():.4f}, "
                f"clean_ce_loss={clean_ce_loss.item():.4f}, "
                f"combined={combined_loss.item():.4f}"
            )

        return combined_loss


   
    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
