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
# See the License for the specific language governing limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .relay_step import (
    acceptance_metrics_to_floats,
    combined_idlm_loss,
    pass2_mask_loss,
    verify_and_build_pass2,
)


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

        self._relay_pass: Optional[int] = None
        self._relay_layout = None
        self._relay_h = None
        self._relay_mask = None
        self._relay_hidden = None
        self._relay_metrics: dict[str, float] = {}

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

    def _model_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Drop collator keys the SDAR forward does not consume."""
        allowed = {"input_ids", "labels", "position_ids", "attention_mask"}
        return {k: v for k, v in inputs.items() if k in allowed and v is not None}

    def _log_idlm(self, task_loss, clean_ce_loss, combined_loss, extra: Optional[dict] = None) -> None:
        alpha = self.finetuning_args.ce_alpha
        log_dict = {
            "train/task_loss": float(task_loss.detach().cpu()),
            "train/clean_ce_loss": float(clean_ce_loss.detach().cpu()),
            "train/combined_loss": float(combined_loss.detach().cpu()),
            "train/alpha": alpha,
        }
        if extra:
            log_dict.update(extra)
        self._relay_metrics.update(log_dict)
        try:
            self.log(log_dict)
        except Exception:
            pass
        logging_steps = getattr(self.args, "logging_steps", 1) or 1
        global_step = getattr(getattr(self, "state", None), "global_step", 0)
        if global_step % logging_steps == 0:
            logger.info_rank0(
                f"Step {global_step}: "
                f"task_loss={log_dict['train/task_loss']:.4f}, "
                f"clean_ce_loss={log_dict['train/clean_ce_loss']:.4f}, "
                f"combined={log_dict['train/combined_loss']:.4f}"
            )

    def _pad_token_id(self) -> Optional[int]:
        proc = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        return getattr(proc, "pad_token_id", None) if proc is not None else None

    def _scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Match HF Trainer.training_step multi-gpu / grad-accum scaling.

        Calling ``super().training_step`` twice would double-count
        ``num_input_tokens_seen`` and apply grad-accum scaling twice, so the
        relay path does two explicit ``compute_loss`` + ``backward`` calls and
        applies this scale itself. Vanilla (``relay_enable=False``) still
        uses the parent ``training_step``.
        """
        if getattr(self.args, "n_gpu", 1) > 1:
            loss = loss.mean()
        accum = getattr(self.args, "gradient_accumulation_steps", 1) or 1
        if accum > 1:
            loss = loss / accum
        return loss

    def _backward(self, loss: torch.Tensor) -> None:
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        # NOTE: The "ar" loss assumes an SDAR model that internally concatenates [noisy|clean],
        # producing hidden_states of length 2*seq_len. This will not work with standard AR models.
        if self.finetuning_args.idlm_loss_type != "ar":
            return super().compute_loss(model, inputs, *args, **kwargs)

        if getattr(self, "_relay_pass", None) == 2:
            return pass2_mask_loss(
                model,
                inputs["input_ids"],
                inputs["labels"],
                self._relay_layout,
                self._relay_h,
                self._relay_mask,
            )

        # Vanilla / pass-1. Keep output_hidden_states=True when relay is off so
        # the graph matches main; with relay on, use last-layer-only capture.
        relay_on = bool(getattr(self.finetuning_args, "relay_enable", False))
        model_inputs = self._model_inputs(inputs)
        extra = {} if relay_on else {"output_hidden_states": True}
        combined, extras = combined_idlm_loss(
            model,
            model_inputs["input_ids"],
            model_inputs["labels"],
            ce_alpha=self.finetuning_args.ce_alpha,
            loss_auto_balance=self.finetuning_args.loss_auto_balance,
            position_ids=model_inputs.get("position_ids"),
            extra_model_kwargs=extra,
        )
        self._relay_hidden = extras["relay_h_last"].detach()
        self._log_idlm(extras["task_loss"], extras["clean_ce_loss"], combined)
        return combined

    @override
    def training_step(self, model, inputs, *args, **kwargs):
        if not getattr(self.finetuning_args, "relay_enable", False):
            # Strictly additive: vanilla path is the parent training_step.
            return super().training_step(model, inputs, *args, **kwargs)

        model.train()
        if hasattr(self, "_prepare_inputs"):
            inputs = self._prepare_inputs(inputs)

        # Pass 1: standard I-DLM loss, capture last-layer h, backward.
        self._relay_pass = 1
        with self.compute_loss_context_manager():
            loss1 = self.compute_loss(model, inputs, *args, **kwargs)
        hidden = self._relay_hidden
        self._backward(self._scale_loss(loss1 * 0.5))

        # Verify (no grad) and pack step-2 canvases.
        base = getattr(model, "module", model)
        block_size = int(base.config.block_size)
        mask_token_id = int(base.config.mask_token_id)
        with torch.no_grad():
            built = verify_and_build_pass2(
                model,
                inputs["input_ids"],
                inputs["labels"],
                hidden,
                block_size=block_size,
                mask_token_id=mask_token_id,
                pad_token_id=self._pad_token_id(),
                rollout_only=bool(getattr(self.finetuning_args, "relay_rollout_only", False)),
                use_regular_causal=bool(getattr(base.config, "use_regular_causal", True)),
            )
        self._relay_layout = built["layout"]
        self._relay_h = built["relay_h"]
        self._relay_mask = built["relay_mask"]
        metrics = acceptance_metrics_to_floats(built["metrics"])
        self._relay_metrics.update(metrics)
        try:
            self.log(metrics)
        except Exception:
            pass

        # Pass 2: mask-CE only on warmstarted canvases (no second clean CE).
        self._relay_pass = 2
        with self.compute_loss_context_manager():
            loss2 = self.compute_loss(model, inputs, *args, **kwargs)
        self._backward(self._scale_loss(loss2 * 0.5))

        self._relay_pass = None
        self._relay_layout = None
        self._relay_h = None
        self._relay_mask = None
        self._relay_hidden = None

        # 0.5*(L1+L2) so logged magnitude matches the vanilla one-pass arm;
        # then apply the same accum scaling HF would have applied.
        return self._scale_loss((loss1.detach() + loss2.detach()) * 0.5)

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
