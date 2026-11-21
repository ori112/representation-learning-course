"""Robustness test (A4 bonus): how much accuracy does a frozen representation lose when
the test images are degraded?

Weakness probed: a representation can score well on clean images and still depend on
fine detail that real-world degradation destroys. A denoising autoencoder is trained to
ignore one kind of damage (patch masking), so this test asks whether that skill carries over
to damage it never saw (noise, blur, low contrast, JPEG) and whether it helps against the
damage it did see.

Why it is a fair test:
- The classifier is trained on CLEAN train features only and then frozen. Nothing adapts
  to the corruption, so any accuracy drop comes from the representation alone.
- Every model sees the identical corrupted images: each (corruption, severity) cell is
  generated with its own fixed seed, independent of the encoder.
- The same linear probe recipe and the same five severities are used for every model.

Run through train.py with mode: robustness.
"""
import io
import json
import math
import os
import statistics

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torchvision.transforms.functional import gaussian_blur

from corruptions import gaussian_noise, patch_masking
from datamodule import NUM_CLASSES, build_datamodule
from evaluate import build_feature_extractor
from evaluation import extract_features, knn_accuracy, top_k_accuracy, train_linear_probe
from model import build_encoder

# Five severities per corruption, mild to harsh. Noise, blur and contrast follow the
# spirit of ImageNet-C. "mask" is the corruption the DAE was trained with (drop rate).
SEVERITIES = {
    "gaussian_noise": [0.04, 0.08, 0.12, 0.18, 0.26],   # sigma on [0, 1] pixels
    "gaussian_blur": [0.5, 1.0, 1.5, 2.0, 2.5],         # sigma in pixels
    "low_contrast": [0.4, 0.3, 0.2, 0.1, 0.05],         # factor on the deviation from the mean
    "jpeg": [25, 18, 12, 8, 5],                         # quality (lower = harsher)
    "patch_mask": [0.1, 0.2, 0.3, 0.4, 0.5],            # share of 8x8 patches removed
}


def blur(images, sigma):
    kernel_size = 2 * math.ceil(3 * sigma) + 1  # covers +-3 sigma
    return gaussian_blur(images, kernel_size=kernel_size, sigma=sigma)


def lower_contrast(images, factor):
    """Pull every pixel toward its image's mean gray level."""
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    return ((images - mean) * factor + mean).clamp(0.0, 1.0)


def jpeg_compress(images, quality):
    """Round-trip through JPEG at the given quality. Done per image on the CPU with PIL."""
    out = []
    for image in (images * 255).round().byte().cpu():
        buffer = io.BytesIO()
        Image.fromarray(image.permute(1, 2, 0).numpy()).save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        out.append(torch.from_numpy(np.array(Image.open(buffer).convert("RGB"))).permute(2, 0, 1))
    return torch.stack(out).float().div(255.0).to(images.device)


CORRUPTION_FUNCTIONS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": blur,
    "low_contrast": lower_contrast,
    "jpeg": jpeg_compress,
    "patch_mask": patch_masking,
}


def corrupted_loader_features(extractor, dataloader, kind, level, seed, device):
    """Extract features of the eval images after corruption. The seed fixes the random
    parts (noise, mask positions), so every encoder is tested on the same damaged images."""
    extractor.eval().to(device)
    generator_state = torch.get_rng_state()
    torch.manual_seed(seed)
    features, labels = [], []
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.float().div(255.0) if images.dtype == torch.uint8 else images
            damaged = CORRUPTION_FUNCTIONS[kind](images, level)
            features.append(extractor(damaged.to(device)).cpu())
            labels.append(batch_labels)
    torch.set_rng_state(generator_state)
    return torch.cat(features), torch.cat(labels)


def run_robustness(config):
    """Train probes on clean train features, then score them on every corrupted eval set."""
    pl.seed_everything(config["seed"], workers=True)  # also fixes random-init encoder weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = NUM_CLASSES[config["dataset"]]

    datamodule = build_datamodule(config)
    datamodule.prepare_data()
    datamodule.setup()
    eval_loader = datamodule.test_dataloader() if config["eval_split"] == "test" else datamodule.val_dataloader()

    encoder = build_encoder(config["latent_dim"])
    if config["encoder_path"] is not None:
        encoder.load_state_dict(torch.load(config["encoder_path"], map_location="cpu"))
    extractor = build_feature_extractor(encoder, config["feature_layer"])

    train_features, train_labels = extract_features(extractor, datamodule.train_eval_dataloader(), device)
    probes = [train_linear_probe(train_features, train_labels, num_classes=num_classes,
                                 epochs=config["probe_epochs"], seed=seed) for seed in config["probe_seeds"]]

    def score(features, labels):
        """Mean probe top-1 over probe seeds, plus k-NN top-1 against the clean train bank."""
        probe_top1 = [top_k_accuracy(probe, features.float(), labels)["top1"] for probe in probes]
        knn_top1 = knn_accuracy(train_features, train_labels, features, labels, num_classes=num_classes)["top1"]
        return statistics.mean(probe_top1), knn_top1

    clean_features, clean_labels = extract_features(extractor, eval_loader, device)
    metrics = {}
    metrics["clean_probe_top1"], metrics["clean_knn_top1"] = score(clean_features, clean_labels)
    for kind, levels in SEVERITIES.items():
        for severity, level in enumerate(levels, start=1):
            seed = 1000 + 10 * list(SEVERITIES).index(kind) + severity  # same for every encoder
            features, labels = corrupted_loader_features(extractor, eval_loader, kind, level, seed, device)
            metrics[f"{kind}_s{severity}_probe_top1"], metrics[f"{kind}_s{severity}_knn_top1"] = score(features, labels)

    os.makedirs(config["results_dir"], exist_ok=True)
    with open(os.path.join(config["results_dir"], f"{config['run_name']}.json"), "w") as file:
        json.dump({"config": config, "metrics": metrics, "severities": SEVERITIES}, file, indent=2)
    return metrics
