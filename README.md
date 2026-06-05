# Thai Image Captioning — Qwen2-VL Fine-Tuning

Fine-tune **Qwen2-VL-2B-Instruct** with QLoRA to generate Thai-language captions for images. Trained on a combined dataset of COCO and IPU24 (food & travel) images with 3 Thai captions per image.

---

## Dataset

| Source | Images | Split |
|--------|--------|-------|
| COCO train2017 | 114,287 | train |
| COCO val2017 | 5,000 | val |
| IPU24 (food + travel) | 28,004 | train |
| IPU24 val | 4,036 | val |
| **Total train** | **142,291** | |
| **Total val** | **9,036** | |

Each image has **3 Thai captions**. One caption is randomly selected per training step as data augmentation.

Annotation format (`capgen_v1.0_train.json`):
```json
{
  "coco/train2017/000000373716.jpg": [
    "ผู้หญิงสวมเสื้อแขนยาวสีขาวและเด็กนั่งเล่นกับสุนัขอยู่ในสวนหย่อม",
    "สาวคนนึงกำลังพาเด็กมานั่งเล่นอยู่ภายในสนามหญ้าพร้อมกับสุนัข",
    "ภาพขาวดำ ผู้หญิงนั่งบนพื้นอุ้มเด็กบนตัก ข้าง ๆ มีหมาสองตัว"
  ],
  "ipu24/train/food/00001.jpg": ["...", "...", "..."]
}
```

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| Base model | `Qwen/Qwen2-VL-2B-Instruct` |
| Vision encoder | Frozen (Qwen2VisionTransformer) |
| Language decoder | Fine-tuned with QLoRA |
| Quantization | 4-bit NF4 (bitsandbytes) |
| LoRA rank | r=16, alpha=32 |
| LoRA targets | `q_proj, k_proj, v_proj, o_proj` |
| Trainable params | ~14M / 2B (0.7%) |
| Training framework | PyTorch Lightning |

**Why Qwen2-VL-2B?** It natively supports Thai (vocab size 152,064), is compact enough to fine-tune on a single GPU with 4-bit quantization, and achieves state-of-the-art vision-language performance.

---

## Project Structure

```
ImageCaptioning/
├── data/
│   ├── dataset.py          # ThaiCaptioningDataset + DataModule
│   └── download_coco.py    # Download COCO train2017 / val2017 images
├── models/
│   └── vlm_captioner.py    # Qwen2VLCaptioner (LightningModule) with QLoRA
├── utils/
│   └── thai_metrics.py     # BLEU-4 / METEOR / CIDEr with Thai word segmentation
├── train.py                # Training entry point
├── evaluate.py             # Offline evaluation
├── inference.py            # Single-image / batch caption generation
├── config.yaml             # All hyperparameters
└── requirements.txt        # Extra dependencies
```

---

## Setup

### 1. Install dependencies

```bash
cd ImageCaptioning
pip install -r requirements.txt
```

`requirements.txt` adds: `peft`, `accelerate`, `bitsandbytes`, `sacrebleu`, `pycocoevalcap`

### 2. Download COCO images

The IPU24 images are already at `/teamspace/studios/this_studio/ipu24/`.  
COCO images must be downloaded (~18 GB total):

```bash
# Download val first (~1 GB, needed for evaluation)
python data/download_coco.py --split val

# Download train in the background (~18 GB)
python data/download_coco.py --split train &

# Check what's already present
python data/download_coco.py --verify-only
```

Images are saved to `/teamspace/studios/this_studio/coco/train2017/` and `.../val2017/`, matching the paths in the JSON annotation files.

---

## Training

### GPU options

| Machine | VRAM | Notes |
|---------|------|-------|
| T4 | 16 GB | Works; reduce batch to 2 if OOM |
| **L4** | **24 GB** | **Recommended** |
| A10G | 24 GB | Works |
| A100 | 40–80 GB | Can increase batch / max_pixels |

### Run training

**Option A — Interactive GPU (switch Studio machine in Lightning AI UI first):**
```bash
python train.py --config config.yaml
```

**Option B — Launch GPU Job from CPU Studio:**
```bash
python train.py --config config.yaml --launch-job --machine L4
```

**Smoke test** (ipu24-only, 500 samples, 1 epoch — verify setup before full run):
```bash
python train.py --config config.yaml --smoke-test
```

**Resume from checkpoint:**
```bash
python train.py --config config.yaml --resume checkpoints/last.ckpt
```

### Staged training (train while COCO downloads)

Set `skip_missing: true` in `config.yaml` to train on ipu24 images only while COCO downloads in the background. Once COCO is ready, set it back to `false` and resume.

### Key hyperparameters (`config.yaml`)

```yaml
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8    # effective batch size = 32
  num_epochs: 3
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  precision: bf16-mixed
```

Expected training time on L4: ~8 hours for 3 epochs over 142K images.

---

## Evaluation

```bash
# Full val set (9,036 images)
python evaluate.py --checkpoint checkpoints/best.ckpt --split val

# Quick estimate (500 images)
python evaluate.py --checkpoint checkpoints/best.ckpt --split val --subset 500

# Greedy decoding (3–4× faster, slightly lower scores)
python evaluate.py --checkpoint checkpoints/best.ckpt --split val --greedy

# Save predictions to JSON
python evaluate.py --checkpoint checkpoints/best.ckpt --split val --output predictions.json
```

### Expected metrics after 3 epochs

| Metric | Expected range |
|--------|---------------|
| BLEU-4 | 20 – 28 |
| METEOR | 22 – 28 |
| CIDEr | 80 – 100 |

> Metrics use **PyThaiNLP word segmentation** (`newmm` engine) since Thai has no spaces between words. Standard whitespace-based tokenization would give misleading scores.

If BLEU-4 < 15 after 3 epochs, try increasing LoRA rank to `r=32` or adding FFN target modules (`gate_proj`, `up_proj`, `down_proj`) in `config.yaml`.

---

## Inference

```bash
# Single image
python inference.py \
  --image /teamspace/studios/this_studio/ipu24/train/food/00001.jpg \
  --checkpoint checkpoints/best.ckpt

# Batch from a file listing image paths (one per line)
python inference.py --image-list my_images.txt --checkpoint checkpoints/best.ckpt

# Base model only (no fine-tuning, for comparison)
python inference.py --image my_photo.jpg --base-only
```

Example output:
```
ipu24/train/food/00001.jpg
  → อาหารไทยรสเลิศวางอยู่บนโต๊ะไม้พร้อมเครื่องปรุงรส
```

---

## Prompt Format

The model is prompted in Thai to encourage Thai output from the first token:

```
<|im_start|>system
คุณเป็นผู้ช่วยที่มีประโยชน์ซึ่งบรรยายภาพเป็นภาษาไทย
<|im_start|>user
[image] บรรยายภาพนี้เป็นภาษาไทย
<|im_start|>assistant
[generated Thai caption]
```

---

## Notes

- **HuggingFace Trainer is not used** — Keras 3.8.0 conflicts with Transformers 4.48.2's Trainer at import time. PyTorch Lightning (2.5.0, already installed) is used instead.
- **Vision encoder is frozen** — only the language decoder's attention layers are trained via LoRA, which keeps training fast and prevents catastrophic forgetting of visual features.
- **Variable-length pixel_values** — Qwen2-VL uses a dynamic resolution mechanism; the collate function uses `torch.cat` (not `torch.stack`) for pixel values.
