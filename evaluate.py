"""
Evaluate a fine-tuned Thai captioning checkpoint.

Usage:
    python evaluate.py --checkpoint checkpoints/best.ckpt --split val
    python evaluate.py --checkpoint checkpoints/best.ckpt --split val --subset 500
    python evaluate.py --checkpoint checkpoints/best.ckpt --split val --greedy
"""

import argparse
import json
import os
import yaml
from typing import List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLProcessor

from models.vlm_captioner import Qwen2VLCaptioner
from utils.thai_metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--subset", type=int, default=None,
                        help="Evaluate on first N images (faster iteration)")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding instead of beam search")
    parser.add_argument("--output", type=str, default=None,
                        help="Save predictions JSON to this path")
    return parser.parse_args()


def build_sample_list(config: dict, split: str):
    json_key = "train_json" if split == "train" else "val_json"
    json_path = config["data"][json_key]
    base_dir = config["data"]["base_dir"]

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for rel_path, captions in raw.items():
        abs_path = os.path.join(base_dir, rel_path)
        if os.path.exists(abs_path):
            samples.append((abs_path, captions))
    return samples


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    processor = Qwen2VLProcessor.from_pretrained(config["model"]["name"])

    print(f"Loading checkpoint: {args.checkpoint}")
    model = Qwen2VLCaptioner.load_from_checkpoint(
        args.checkpoint,
        config=config,
        strict=False,
    )
    model.eval()

    samples = build_sample_list(config, args.split)
    if args.subset:
        samples = samples[: args.subset]
    print(f"Evaluating {len(samples)} images ...")

    beam_size = 1 if args.greedy else config["inference"]["beam_size"]
    max_new_tokens = config["inference"]["max_new_tokens"]
    rep_penalty = config["inference"]["repetition_penalty"]

    predictions: List[str] = []
    references: List[List[str]] = []

    for abs_path, captions in tqdm(samples, desc="Generating"):
        image = Image.open(abs_path).convert("RGB")
        caption = model.generate_caption(
            processor=processor,
            image=image,
            beam_size=beam_size,
            max_new_tokens=max_new_tokens,
            repetition_penalty=rep_penalty,
        )
        predictions.append(caption)
        references.append(captions)

    metrics = compute_metrics(predictions, references)
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k.upper():8s}: {v:.2f}")

    if args.output:
        results = [
            {"image": s[0], "prediction": p, "references": r}
            for s, p, r in zip(samples, predictions, references)
        ]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)
        print(f"\nSaved predictions to {args.output}")


if __name__ == "__main__":
    main()
