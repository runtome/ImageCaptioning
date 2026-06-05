"""
Thai image captioning training script.

Local GPU run (after switching Studio machine to T4 / L4):
    python train.py --config config.yaml

Launch GPU job from CPU Studio via lightning_sdk:
    python train.py --config config.yaml --launch-job --machine L4

Resume from checkpoint:
    python train.py --config config.yaml --resume checkpoints/last.ckpt

Smoke test on ipu24-only data (skip COCO images):
    python train.py --config config.yaml --smoke-test
"""

import argparse
import os
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import Qwen2VLProcessor

from data.dataset import ThaiCaptioningDataModule
from models.vlm_captioner import Qwen2VLCaptioner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick smoke test: 1 epoch, skip_missing=true, 500 samples",
    )
    parser.add_argument(
        "--launch-job",
        action="store_true",
        help="Launch as a Lightning AI GPU Job instead of running locally",
    )
    parser.add_argument(
        "--machine",
        default="L4",
        choices=["T4", "L4", "A10G", "L40S", "A100_X_8"],
    )
    return parser.parse_args()


def launch_lightning_job(args) -> None:
    try:
        from lightning_sdk import Studio, Machine
    except ImportError:
        raise SystemExit(
            "lightning_sdk not installed. Run: pip install lightning_sdk"
        )

    machine_map = {
        "T4": Machine.T4,
        "L4": Machine.L4,
        "A10G": Machine.A10G,
        "L40S": Machine.L40S,
        "A100_X_8": Machine.A100_X_8,
    }

    studio = Studio()
    studio.start()
    job = studio.run(
        "cd /teamspace/studios/this_studio/ImageCaptioning && "
        "pip install -r requirements.txt -q && "
        f"python train.py --config {args.config}",
        machine=machine_map[args.machine],
    )
    print(f"Job launched on {args.machine}: {job}")


def run_training(config: dict, resume_checkpoint=None) -> None:
    pl.seed_everything(42, workers=True)

    processor = Qwen2VLProcessor.from_pretrained(config["model"]["name"])

    datamodule = ThaiCaptioningDataModule(config, processor)
    model = Qwen2VLCaptioner(config)

    cfg_t = config["training"]
    cfg_log = config["logging"]

    os.makedirs(cfg_t["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg_log["tensorboard_dir"], exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg_t["checkpoint_dir"],
        filename="qwen2vl-thai-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=cfg_t["save_top_k"],
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tb_logger = TensorBoardLogger(
        save_dir=cfg_log["tensorboard_dir"],
        name="thai_captioning",
    )

    loggers = [tb_logger]
    if cfg_log.get("wandb_project"):
        from pytorch_lightning.loggers import WandbLogger
        loggers.append(WandbLogger(project=cfg_log["wandb_project"]))

    trainer = pl.Trainer(
        max_epochs=cfg_t["num_epochs"],
        accumulate_grad_batches=cfg_t["gradient_accumulation_steps"],
        precision=cfg_t["precision"],
        val_check_interval=cfg_t["val_check_interval"],
        gradient_clip_val=cfg_t["max_grad_norm"],
        callbacks=[checkpoint_cb, lr_monitor],
        logger=loggers,
        log_every_n_steps=cfg_log["log_every_n_steps"],
        deterministic=False,
        enable_progress_bar=True,
        devices=1,
        accelerator="gpu",
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_checkpoint)
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.smoke_test:
        config["data"]["skip_missing"] = True
        config["training"]["num_epochs"] = 1
        config["training"]["val_check_interval"] = 0.5
        # Limit dataset size for quick iteration
        config["data"]["_max_samples"] = 500
        print("[smoke-test] Running with 500 samples, skip_missing=true, 1 epoch")

    if args.launch_job:
        launch_lightning_job(args)
        return

    run_training(config, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()
