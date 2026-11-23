"""Launch one experiment: python train.py --config configs/<name>.yaml

One config file = one run. The run writes, under its run_name:
  outputs/checkpoints/<run_name>/last.ckpt   resume point (Colab can disconnect); deleted once the run is done
  outputs/tb_logs/<run_name>/                TensorBoard logs
  outputs/encoders/<run_name>.pt             the encoder weights, for evaluation (pre-training runs)
  results/<run_name>.json                    final metrics + the full config

With early_stopping_patience set, training stops when the validation metric (val_top1 for fine-tuning,
val_recon_loss for the autoencoder) has not improved for that many epochs, and the best epoch is kept.
"""
import json
import os
import shutil
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from callbacks import ColabProgressBar, KNNMonitor
from config import load_config
from datamodule import build_datamodule
from evaluation import knn_by_layer, run_evaluation
from robustness import run_robustness
from utils import build_model, parse_args

torch.set_float32_matmul_precision("medium")  # faster matmuls on modern GPUs, silences a Lightning warning

# What the stop rule watches per mode: (validation metric, direction of "better").
STOP_METRIC = {"pretrain_dae": ("val_recon_loss", "min"), "finetune": ("val_top1", "max")}


def run(config):
    """Train one model from a config dict and save its encoder and metrics."""
    pl.seed_everything(config["seed"], workers=True)  # same init + data order -> fair comparison

    run_name = config["run_name"]
    checkpoint_dir = os.path.join(config["output_dir"], "checkpoints", run_name)
    use_gpu = torch.cuda.is_available()
    precision = ("16-mixed" if use_gpu else "32-true") if config["precision"] == "auto" else config["precision"]

    datamodule = build_datamodule(config)
    model = build_model(config)

    callbacks = [
        # last.ckpt is what resume loads. save_top_k=0 skips the extra per-epoch copy.
        ModelCheckpoint(dirpath=checkpoint_dir, save_last=True, save_top_k=0),
        ColabProgressBar(refresh_rate=20),
    ]
    if config["mode"] == "pretrain_dae":
        callbacks.append(KNNMonitor(every_n_epochs=config["knn_every_n_epochs"], feature_layer=config["feature_layer"]))
    best_checkpoint = None
    if config["early_stopping_patience"] is not None:
        monitor, direction = STOP_METRIC[config["mode"]]
        # Stop when the validation metric has not improved for `patience` epochs, and keep the best epoch.
        callbacks.append(EarlyStopping(monitor=monitor, mode=direction, patience=config["early_stopping_patience"],
                                      min_delta=config["early_stopping_min_delta"]))
        best_checkpoint = ModelCheckpoint(dirpath=os.path.join(checkpoint_dir, "best"), filename="best",
                                          monitor=monitor, mode=direction, save_top_k=1)
        callbacks.append(best_checkpoint)

    trainer = pl.Trainer(
        logger=TensorBoardLogger(os.path.join(config["output_dir"], "tb_logs"), name=run_name, default_hp_metric=False),
        accelerator="auto",
        devices=1,
        max_epochs=config["max_epochs"],
        precision=precision,
        fast_dev_run=config["fast_dev_run"],
        log_every_n_steps=10,
        callbacks=callbacks,
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
    metrics["epochs_trained"] = trainer.current_epoch
    if best_checkpoint is not None:
        # Go back to the best epoch: its weights are the model, and its validation numbers are the ones reported.
        checkpoint = torch.load(best_checkpoint.best_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        metrics["best_epoch"] = checkpoint["epoch"] + 1
        trainer.validate(model, datamodule=datamodule, verbose=False)
        metrics.update({name: value.item() for name, value in trainer.callback_metrics.items()})
        if config["mode"] == "pretrain_dae":
            # KNNMonitor only runs every few epochs, so measure the best-epoch encoder directly.
            device = "cuda" if use_gpu else "cpu"
            for layer, accuracy in knn_by_layer(model.encoder, datamodule, device).items():
                metrics[f"val_knn_top1_{layer}"] = accuracy["top1"]
                if layer == config["feature_layer"]:
                    metrics["val_knn_top1"], metrics["val_knn_top5"] = accuracy["top1"], accuracy["top5"]
    if config["mode"] == "finetune" and config["eval_split"] == "test":
        trainer.test(model, datamodule=datamodule)  # the one allowed look at the test set
        # trainer.test replaces callback_metrics, so merge instead of overwrite to keep the val numbers.
        metrics.update({name: value.item() for name, value in trainer.callback_metrics.items()})

    if config["mode"] == "pretrain_dae" and config["save_encoder"]:
        # Keep only the encoder: the decoder is discarded after training.
        os.makedirs(os.path.join(config["output_dir"], "encoders"), exist_ok=True)
        torch.save(model.encoder.state_dict(), os.path.join(config["output_dir"], "encoders", f"{run_name}.pt"))

    os.makedirs(config["results_dir"], exist_ok=True)
    with open(os.path.join(config["results_dir"], f"{run_name}.json"), "w") as file:
        json.dump({"config": config, "metrics": metrics, "train_seconds": round(train_seconds, 1),
                   "device": torch.cuda.get_device_name(0) if use_gpu else "cpu",
                   "precision": precision, "resumed": resume_from is not None}, file, indent=2)
    # Checkpoints are only for resuming an interrupted run. Once the results are written they are dead weight
    # (a DAE checkpoint holds the weights and the Adam state, hundreds of MB), so free the space, notably on Drive.
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return metrics


if __name__ == "__main__":
    config = load_config(parse_args().config)
    if config["mode"] == "evaluate":
        final_metrics = run_evaluation(config)
    elif config["mode"] == "robustness":
        final_metrics = run_robustness(config)
    else:
        final_metrics = run(config)
    print("Final metrics:", final_metrics)
