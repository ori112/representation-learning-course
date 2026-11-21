"""Evaluate one encoder without training it: python train.py --config configs/<eval>.yaml (mode: evaluate).

Measures the representation with the protocol of the assignment: weighted cosine
k-NN and a linear probe, both on frozen features. encoder_path = None evaluates a
random-init encoder, the "no learning" baseline every trained model must beat.
"""
import json
import os
import statistics

import pytorch_lightning as pl
import torch
from torch import nn

from datamodule import NUM_CLASSES, build_datamodule
from evaluation import extract_features, knn_accuracy, linear_probe_accuracy
from model import build_encoder


def build_feature_extractor(encoder, feature_layer):
    """Return the module whose output is the representation h that gets evaluated.

    "bottleneck" is the whole encoder (the 64-d Linear output). "conv" stops after the
    second ReLU and average-pools 16x16 to 4x4, giving 64*4*4 = 1024 numbers. Pooling
    keeps the probe small: the raw map has 16384 numbers per image.
    """
    if feature_layer == "bottleneck":
        return encoder
    if feature_layer == "conv":
        return nn.Sequential(encoder[:4], nn.AdaptiveAvgPool2d(4), nn.Flatten())
    raise ValueError(f"Unknown feature_layer: {feature_layer}")


def run_evaluation(config):
    """Evaluate an encoder on the validation or test split, save results/<run_name>.json."""
    pl.seed_everything(config["seed"], workers=True)  # also fixes the random-init weights
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datamodule = build_datamodule(config)
    datamodule.prepare_data()
    datamodule.setup()

    encoder = build_encoder(config["latent_dim"])
    if config["encoder_path"] is not None:
        encoder.load_state_dict(torch.load(config["encoder_path"], map_location="cpu"))
    extractor = build_feature_extractor(encoder, config["feature_layer"])

    eval_loader = datamodule.test_dataloader() if config["eval_split"] == "test" else datamodule.val_dataloader()
    train_features, train_labels = extract_features(extractor, datamodule.train_eval_dataloader(), device)
    eval_features, eval_labels = extract_features(extractor, eval_loader, device)

    num_classes = NUM_CLASSES[config["dataset"]]
    knn = knn_accuracy(train_features, train_labels, eval_features, eval_labels, num_classes=num_classes)
    probes = [linear_probe_accuracy(train_features, train_labels, eval_features, eval_labels,
                                    num_classes=num_classes, epochs=config["probe_epochs"], seed=seed)
              for seed in config["probe_seeds"]]
    probe_top1 = [probe["top1"] for probe in probes]
    metrics = {
        "knn_top1": knn["top1"], "knn_top5": knn["top5"],
        "probe_top1_mean": statistics.mean(probe_top1),
        "probe_top1_std": statistics.pstdev(probe_top1),  # 0 with a single seed
        "probe_top5_mean": statistics.mean(probe["top5"] for probe in probes),
        "feature_dim": train_features.shape[1],
    }

    os.makedirs(config["results_dir"], exist_ok=True)
    with open(os.path.join(config["results_dir"], f"{config['run_name']}.json"), "w") as file:
        json.dump({"config": config, "metrics": metrics}, file, indent=2)
    return metrics
