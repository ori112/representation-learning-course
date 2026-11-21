"""Launch one experiment: python train.py --config configs/<name>.yaml

One config file = one run. The run writes, under its run_name:
  outputs/checkpoints/<run_name>/last.ckpt   resume point (Colab can disconnect)
  outputs/tb_logs/<run_name>/                TensorBoard logs
  outputs/encoders/<run_name>.pt             the encoder weights, for evaluation
  results/<run_name>.json                    final metrics + the full config
"""
import argparse
import json
import os
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from callbacks import ColabProgressBar, KNNMonitor
from config import load_config
from datamodule import build_datamodule
from evaluate import run_evaluation
from utils import build_model

torch.set_float32_matmul_precision("medium")  # faster matmuls on modern GPUs, silences a Lightning warning


def run(config):
    """Train one model from a config dict and save its encoder and metrics."""
    pl.seed_everything(config["seed"], workers=True)  # same init + data order -> fair comparison

    run_name = config["run_name"]
    checkpoint_dir = os.path.join(config["output_dir"], "checkpoints", run_name)
    use_gpu = torch.cuda.is_available()
    precision = ("16-mixed" if use_gpu else "32-true") if config["precision"] == "auto" else config["precision"]

    datamodule = build_datamodule(config)
    model = build_model(config)
    trainer = pl.Trainer(
        logger=TensorBoardLogger(os.path.join(config["output_dir"], "tb_logs"), name=run_name, default_hp_metric=False),
        accelerator="auto",
        devices=1,
        max_epochs=config["max_epochs"],
        precision=precision,
        fast_dev_run=config["fast_dev_run"],
        log_every_n_steps=10,
        callbacks=[
            # last.ckpt is what resume loads. save_top_k=0 skips the extra per-epoch copy.
            ModelCheckpoint(dirpath=checkpoint_dir, save_last=True, save_top_k=0),
            ColabProgressBar(refresh_rate=20),
        ] + ([KNNMonitor(every_n_epochs=config["knn_every_n_epochs"])] if config["mode"] == "pretrain_dae" else []),
    )

    resume_from = os.path.join(checkpoint_dir, "last.ckpt")
    resume_from = resume_from if config["resume"] and os.path.exists(resume_from) else None
    # Build the dataset cache first. trainer.fit would do it too, but then the first
    # run in a fresh Colab session would count the download as training time.
    datamodule.prepare_data()
    start_time = time.time()
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_from)
    train_seconds = time.time() - start_time

    if config["fast_dev_run"]:
        return {}  # wiring test only: nothing worth saving

    metrics = {name: value.item() for name, value in trainer.callback_metrics.items()}
    if config["mode"] == "finetune" and config["eval_split"] == "test":
        trainer.test(model, datamodule=datamodule)  # the one allowed look at the test set
        # trainer.test replaces callback_metrics, so merge instead of overwrite to keep the val numbers.
        metrics.update({name: value.item() for name, value in trainer.callback_metrics.items()})

    if config["mode"] == "pretrain_dae":
        # Keep only the encoder: the decoder is discarded after training.
        os.makedirs(os.path.join(config["output_dir"], "encoders"), exist_ok=True)
        torch.save(model.encoder.state_dict(), os.path.join(config["output_dir"], "encoders", f"{run_name}.pt"))

    os.makedirs(config["results_dir"], exist_ok=True)
    with open(os.path.join(config["results_dir"], f"{run_name}.json"), "w") as file:
        json.dump({"config": config, "metrics": metrics, "train_seconds": round(train_seconds, 1),
                   "device": torch.cuda.get_device_name(0) if use_gpu else "cpu",
                   "precision": precision, "resumed": resume_from is not None}, file, indent=2)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one experiment from a YAML config.")
    parser.add_argument("--config", required=True, help="path to a YAML file that overrides config.DEFAULTS")
    config = load_config(parser.parse_args().config)
    final_metrics = run_evaluation(config) if config["mode"] == "evaluate" else run(config)
    print("Final metrics:", final_metrics)
