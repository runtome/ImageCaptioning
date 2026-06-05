from typing import Any, Dict, List, Optional

import torch
import pytorch_lightning as pl
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


class Qwen2VLCaptioner(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.cfg = config
        self.model = self._load_model()

        # Buffers for computing generation metrics at epoch end
        self._val_preds: List[str] = []
        self._val_refs: List[List[str]] = []

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _load_model(self) -> Qwen2VLForConditionalGeneration:
        cfg_m = self.cfg["model"]

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg_m["load_in_4bit"],
            bnb_4bit_quant_type=cfg_m["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=cfg_m["bnb_4bit_use_double_quant"],
        )

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg_m["name"],
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.cfg["training"]["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        for param in model.visual.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=cfg_m["lora_r"],
            lora_alpha=cfg_m["lora_alpha"],
            lora_dropout=cfg_m["lora_dropout"],
            target_modules=cfg_m["lora_target_modules"],
            bias=cfg_m["lora_bias"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            labels=batch["labels"],
        )
        self.log("train/loss", outputs.loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return outputs.loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                labels=batch["labels"],
            )
        self.log("val/loss", outputs.loss, on_epoch=True, prog_bar=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        try:
            from bitsandbytes.optim import AdamW8bit as Optimizer
        except ImportError:
            from torch.optim import AdamW as Optimizer

        cfg_t = self.cfg["training"]
        optimizer = Optimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg_t["learning_rate"],
            weight_decay=cfg_t["weight_decay"],
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * cfg_t["warmup_ratio"])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ------------------------------------------------------------------
    # Generation helper (used by evaluate.py and inference.py)
    # ------------------------------------------------------------------

    def generate_caption(
        self,
        processor: Qwen2VLProcessor,
        image,
        beam_size: int = 4,
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.2,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": "คุณเป็นผู้ช่วยที่มีประโยชน์ซึ่งบรรยายภาพเป็นภาษาไทย",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "บรรยายภาพนี้เป็นภาษาไทย"},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=beam_size,
                repetition_penalty=repetition_penalty,
            )

        # Decode only the newly generated tokens
        new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        caption = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        return caption.strip()
