"""
Single-image Thai caption generation.

Usage:
    # From a fine-tuned checkpoint:
    python inference.py --image /path/to/image.jpg --checkpoint checkpoints/best.ckpt

    # From the base model (no fine-tuning):
    python inference.py --image /path/to/image.jpg --base-only

    # Batch mode from a text file listing image paths:
    python inference.py --image-list images.txt --checkpoint checkpoints/best.ckpt
"""

import argparse
import os
import yaml

import torch
from PIL import Image
from transformers import Qwen2VLProcessor

from models.vlm_captioner import Qwen2VLCaptioner


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image file")
    group.add_argument("--image-list", type=str, help="Text file with one image path per line")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned .ckpt file")
    parser.add_argument("--base-only", action="store_true",
                        help="Run the base model without a fine-tuned checkpoint")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def load_model(args, config: dict) -> tuple:
    processor = Qwen2VLProcessor.from_pretrained(config["model"]["name"])

    if args.base_only:
        # Load base model without QLoRA for quick testing
        from transformers import Qwen2VLForConditionalGeneration
        model_raw = Qwen2VLForConditionalGeneration.from_pretrained(
            config["model"]["name"],
            torch_dtype=torch.bfloat16,
            device_map={"": 0} if torch.cuda.is_available() else "cpu",
        )
        # Wrap in a thin object that exposes generate_caption
        class _Wrapper:
            def __init__(self, m):
                self._m = m
                self.model = m
            def generate_caption(self, processor, image, beam_size=4,
                                  max_new_tokens=128, repetition_penalty=1.2):
                from models.vlm_captioner import Qwen2VLCaptioner
                # Use the static generation logic
                messages = [
                    {"role": "system",
                     "content": "คุณเป็นผู้ช่วยที่มีประโยชน์ซึ่งบรรยายภาพเป็นภาษาไทย"},
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "บรรยายภาพนี้เป็นภาษาไทย"},
                    ]},
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(images=image, text=text, return_tensors="pt")
                device = next(self._m.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    ids = self._m.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_beams=beam_size,
                        repetition_penalty=repetition_penalty,
                    )
                new_ids = ids[:, inputs["input_ids"].shape[1]:]
                return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        return _Wrapper(model_raw), processor

    if args.checkpoint is None:
        raise ValueError("Provide --checkpoint or use --base-only")

    model = Qwen2VLCaptioner.load_from_checkpoint(
        args.checkpoint,
        config=config,
        strict=False,
    )
    model.eval()
    return model, processor


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model, processor = load_model(args, config)

    beam_size = args.beam_size or config["inference"]["beam_size"]
    max_new_tokens = args.max_new_tokens or config["inference"]["max_new_tokens"]
    rep_penalty = config["inference"]["repetition_penalty"]

    image_paths = []
    if args.image:
        image_paths = [args.image]
    else:
        with open(args.image_list, "r") as f:
            image_paths = [line.strip() for line in f if line.strip()]

    for path in image_paths:
        if not os.path.exists(path):
            print(f"[skip] Not found: {path}")
            continue
        image = Image.open(path).convert("RGB")
        caption = model.generate_caption(
            processor=processor,
            image=image,
            beam_size=beam_size,
            max_new_tokens=max_new_tokens,
            repetition_penalty=rep_penalty,
        )
        print(f"{path}\n  → {caption}\n")


if __name__ == "__main__":
    main()
