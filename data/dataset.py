import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule


class ThaiCaptioningDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        base_dir: str,
        processor,
        split: str = "train",
        skip_missing: bool = False,
        min_pixels: int = 256,
        max_pixels: int = 1280,
        max_samples: Optional[int] = None,
    ) -> None:
        self.processor = processor
        self.split = split
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples: List[Tuple[str, List[str]]] = []
        skipped = 0
        for rel_path, captions in raw.items():
            abs_path = os.path.join(base_dir, rel_path)
            if skip_missing and not os.path.exists(abs_path):
                skipped += 1
                continue
            self.samples.append((abs_path, captions))

        if skipped:
            print(f"[{split}] Skipped {skipped} entries with missing images.")
        print(f"[{split}] Loaded {len(self.samples)} samples from {json_path}")

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        abs_path, captions = self.samples[idx]

        # Random caption selection for training augmentation
        caption = random.choice(captions) if self.split == "train" else captions[0]

        image = Image.open(abs_path).convert("RGB")

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

        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + caption + "<|im_end|>"

        inputs = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding=False,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()

        # Mask everything up to (and including) the assistant turn start.
        # Find the last <|im_start|> token — that's where the assistant response begins.
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            last_im_start = positions[-1].item()
            # Mask: <|im_start|> + "assistant" + "\n" = 3 tokens
            labels[: last_im_start + 3] = -100
        else:
            # Fallback: mask first half if special token not found
            labels[: len(input_ids) // 2] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"],       # (num_patches, C, pH, pW)
            "image_grid_thw": inputs["image_grid_thw"],   # (1, 3)
            "labels": labels,
        }


def qwen_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate variable-length sequences for Qwen2-VL.
    pixel_values: concatenate along patch dim (not stack)
    image_grid_thw: concatenate
    input_ids / attention_mask / labels: right-pad to max length
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    padded_ids, padded_masks, padded_labels = [], [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        padded_ids.append(F.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        padded_masks.append(F.pad(item["attention_mask"], (0, pad_len), value=0))
        padded_labels.append(F.pad(item["labels"], (0, pad_len), value=-100))

    return {
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(padded_masks),
        "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
        "image_grid_thw": torch.cat([item["image_grid_thw"] for item in batch], dim=0),
        "labels": torch.stack(padded_labels),
    }


class ThaiCaptioningDataModule(LightningDataModule):
    def __init__(self, config: dict, processor) -> None:
        super().__init__()
        self.config = config
        self.processor = processor
        self.train_dataset: Optional[ThaiCaptioningDataset] = None
        self.val_dataset: Optional[ThaiCaptioningDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        cfg_data = self.config["data"]
        cfg_model = self.config["model"]

        shared = dict(
            base_dir=cfg_data["base_dir"],
            processor=self.processor,
            skip_missing=cfg_data.get("skip_missing", False),
            min_pixels=cfg_model.get("min_pixels", 256),
            max_pixels=cfg_model.get("max_pixels", 1280),
        )

        if stage in ("fit", None):
            self.train_dataset = ThaiCaptioningDataset(
                json_path=cfg_data["train_json"],
                split="train",
                **shared,
            )

        if stage in ("fit", "validate", None):
            self.val_dataset = ThaiCaptioningDataset(
                json_path=cfg_data["val_json"],
                split="val",
                **shared,
            )

    def _collate(self, batch):
        pad_id = self.processor.tokenizer.pad_token_id or 0
        return qwen_collate_fn(batch, pad_token_id=pad_id)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["per_device_train_batch_size"],
            shuffle=True,
            num_workers=self.config["data"].get("num_workers", 4),
            pin_memory=self.config["data"].get("pin_memory", True),
            collate_fn=self._collate,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["training"]["per_device_val_batch_size"],
            shuffle=False,
            num_workers=self.config["data"].get("num_workers", 4),
            pin_memory=self.config["data"].get("pin_memory", True),
            collate_fn=self._collate,
        )
