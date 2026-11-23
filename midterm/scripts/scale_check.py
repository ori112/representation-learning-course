"""Why does the DAE's linear probe collapse under damage while its k-NN barely moves?

Hypothesis: damage shifts the feature vectors (mean and length) away from the clean
train features the probe saw. A linear probe depends on that position, while cosine k-NN
uses only the direction. This script measures the shift on the validation split:
  - norm_ratio:  mean ||h(corrupted)|| / mean ||h(clean)||
  - mean_shift:  average over dims of |mean(corrupted) - mean(clean)| / std(clean)
  - cosine_to_clean: cosine between each image's corrupted and clean feature (direction kept?)
Run: uv run python -m scripts.scale_check   (about a minute on CPU)
"""
import csv

import torch
import torch.nn.functional as F

from datamodule import TinyImageNetDataModule
from evaluation import extract_features
from scripts.interpret import load_extractor
from robustness import corrupted_loader_features

CASES = [("gaussian_noise", 2, 0.08), ("patch_mask", 1, 0.1), ("gaussian_blur", 3, 1.5), ("jpeg", 3, 12)]


def main():
    datamodule = TinyImageNetDataModule()
    datamodule.setup()
    rows = []
    for name in ("dae", "ae", "random"):
        extractor = load_extractor(name)
        clean, _ = extract_features(extractor, datamodule.val_dataloader(), "cpu")
        for kind, severity, level in CASES:
            damaged, _ = corrupted_loader_features(extractor, datamodule.val_dataloader(), kind, level,
                                                   seed=1000 + severity, device="cpu")
            rows.append([name, f"{kind}_s{severity}",
                         round((damaged.norm(dim=1).mean() / clean.norm(dim=1).mean()).item(), 3),
                         round(((damaged.mean(0) - clean.mean(0)).abs() / clean.std(0).clamp_min(1e-6)).mean().item(), 3),
                         round(F.cosine_similarity(damaged, clean, dim=1).mean().item(), 3)])
            print(rows[-1])
    with open("summaries/robustness_feature_shift.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "corruption", "norm_ratio", "mean_shift_in_std", "cosine_to_clean"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
