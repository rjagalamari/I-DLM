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
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union
# For verifying it..
from .fused_verify_kernel import fused_spec_verify_from_logits

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
         ## relaying the hidden state for the second pass...
        if (self.finetuning_args.relay_enable and getattr(self, "_relay_pass", None) == 1):
            self._relay_hidden = outputs["hidden_states"][-1].detach()
        task_loss = outputs["loss"]  # CE on all positions (noisy + clean)


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
            outputs["hidden_states"][-1][:, seq_len : seq_len + seq_len, :]
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

    ## drafting the pass1 hidden states and verifying...
    @torch.no_grad()
    def _get_relay_drafts(self, model, noisy_hidden):
      base = getattr(model, "module", model)

      logits = base.lm_head(
          noisy_hidden.to(base.lm_head.weight.dtype)
      )

      return logits.argmax(dim=-1)

    # Verify the drafts..
    @torch.no_grad()
    def _verify_relay_drafts(
      self, model, noisy_hidden, clean_hidden, inputs
    ):
        base = getattr(model, "module", model)
        device = noisy_hidden.device

        labels = inputs["labels"].to(device)
        input_ids = inputs["input_ids"].to(device)

        pad_token_id = self.processing_class.pad_token_id
        token_valid = labels.ne(-100)

        if pad_token_id is not None:
            token_valid = token_valid & input_ids.ne(pad_token_id)
        
        valid_positions = torch.zeros_like(token_valid)
        valid_positions[:, :-1] = token_valid[:, 1:]

        drafts = torch.full_like(labels, -100)
        accepted = torch.zeros_like(valid_positions)

        if not valid_positions.any():
              return drafts, accepted, valid_positions
        # noisy logits..
        lm_dtype = base.lm_head.weight.dtype
        noisy_logits = base.lm_head(noisy_hidden[valid_positions].to(lm_dtype))
        selected_drafts = noisy_logits.argmax(dim=-1)
        #clean logits..
        clean_logits = base.lm_head(clean_hidden[valid_positions].to(lm_dtype))
        # Verify the drafts using fused kernel..
        selected_accepted, _ = fused_spec_verify_from_logits(
              clean_logits, noisy_logits, selected_drafts, temperature=1.0, alpha=1.0
          )

        drafts[valid_positions] = selected_drafts
        accepted[valid_positions] = selected_accepted.bool()
        ## making the block level acceptance mask for the next pass...
        block_size = base.config.block_size
        keep = torch.zeros_like(accepted)
        for batch_idx in range(accepted.shape[0]):
              for start in range(0, accepted.shape[1], block_size):
                    end = min(start + block_size, accepted.shape[1])
                    for pos in range(start, end):






    # Two pass training step...
    @override
    def training_step(self, model, inputs, *args, **kwargs):
      if not self.finetuning_args.relay_enable:
          return super().training_step(model, inputs, *args, **kwargs)

      self._relay_hidden = None

      self._relay_pass = 1
      loss1 = super().training_step(model, inputs, *args, **kwargs)
      # Hidden states
      hidden = self._relay_hidden
      seq_len = hidden.shape[1] // 2

      noisy_hidden = hidden[:, :seq_len, :]
      clean_hidden = hidden[:, seq_len:, :]
      # Generate drafts and verfiy..
      drafts, accepted = self._verify_relay_drafts(
        model, noisy_hidden, clean_hidden, inputs
      )
      self._relay_pass = 2
      loss2 = super().training_step(model, inputs, *args, **kwargs)

      self._relay_pass = None
      self._relay_hidden = None

      return (loss1 + loss2) / 2
    


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
