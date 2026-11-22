"""A5 evidence: per-class accuracy, t-SNE, retrieval grids, and a probe sanity check.

Run after the encoders exist (outputs/encoders/final_dae.pt, final_ae.pt):
    uv run python -m scripts.interpret   (from the midterm folder)
Everything is computed on the validation hold-out, never on the test split.
Outputs go to figures/ (images) and summaries/ (small CSVs).
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

from datamodule import TinyImageNetDataModule
from model import build_encoder, build_feature_extractor
from evaluation import extract_features, knn_accuracy, train_linear_probe, top_k_accuracy


ENCODERS = {"dae": "outputs/encoders/final_dae.pt", "ae": "outputs/encoders/final_ae.pt", "random": None}
LATENT_DIM = 512
TSNE_CLASSES = 20          # first 20 class ids: arbitrary but fixed, not cherry-picked
RETRIEVAL_QUERIES = 4      # first query image of the first 4 classes: small enough for the report
RETRIEVAL_MODELS = ("dae", "random")  # the DAE against the no-learning floor
FEATURE_CACHE = "outputs/a5_features.pt"  # gitignored; extraction is the slow part on CPU
RETRIEVAL_NEIGHBORS = 6


def load_extractor(name):
    """Pooled-conv feature extractor (the chosen h) for dae / ae / random."""
    torch.manual_seed(42)  # same seed as evaluate.py, so "random" is the same random encoder
    encoder = build_encoder(LATENT_DIM)
    if ENCODERS[name] is not None:
        encoder.load_state_dict(torch.load(ENCODERS[name], map_location="cpu"))
    return build_feature_extractor(encoder, "conv")


def per_class_knn(bank_features, bank_labels, val_features, val_labels, num_classes=200):
    """k-NN top-1 per class: predict every val image, then average within each class."""
    correct = torch.zeros(len(val_labels))
    for start in range(0, len(val_labels), 1000):  # same weighted cosine vote as knn_accuracy
        query = F.normalize(val_features[start:start + 1000], dim=1)
        similarity = query @ F.normalize(bank_features, dim=1).T
        top_sim, top_idx = similarity.topk(20, dim=1)
        votes = torch.zeros(len(query), num_classes)
        votes.scatter_add_(1, bank_labels[top_idx], (top_sim / 0.07).exp())
        correct[start:start + 1000] = (votes.argmax(1) == val_labels[start:start + 1000]).float()
    return torch.stack([correct[val_labels == c].mean() for c in range(num_classes)])


def plot_tsne(features_by_model, labels, save_path):
    """t-SNE of 20 classes for each encoder, same images, same classes."""
    keep = labels < TSNE_CLASSES
    fig, axes = plt.subplots(1, len(features_by_model), figsize=(5 * len(features_by_model), 4.8), squeeze=False)
    axes = axes[0]
    for ax, (name, features) in zip(axes, features_by_model.items()):
        embedded = TSNE(n_components=2, init="pca", perplexity=30, random_state=42).fit_transform(
            F.normalize(features[keep], dim=1).numpy())
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=labels[keep].numpy(), cmap="tab20", s=6)
        ax.set_title(f"{name}: t-SNE, {TSNE_CLASSES} classes")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(scatter, ax=axes, label="class id", shrink=0.8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_retrieval(images, labels, bank_images, bank_labels, features_by_model, bank_by_model, save_path):
    """For fixed query images, show the nearest train images per encoder.
    Green frame = same class as the query, red = different class."""
    features_by_model = {name: f for name, f in features_by_model.items() if name in RETRIEVAL_MODELS}
    first_seen = {}
    for index, label in enumerate(labels.tolist()):
        if label < RETRIEVAL_QUERIES and label not in first_seen:
            first_seen[label] = index
    query_ids = list(first_seen.values())
    fig, axes = plt.subplots(len(query_ids) * len(features_by_model), 1 + RETRIEVAL_NEIGHBORS,
                             figsize=(1.3 * (1 + RETRIEVAL_NEIGHBORS), 1.3 * len(query_ids) * len(features_by_model)))
    row = 0
    for query_id in query_ids:
        for name, features in features_by_model.items():
            sim = F.normalize(features[query_id:query_id + 1], dim=1) @ F.normalize(bank_by_model[name], dim=1).T
            neighbors = sim.topk(RETRIEVAL_NEIGHBORS, dim=1).indices[0]
            axes[row][0].imshow(images[query_id].permute(1, 2, 0).numpy())
            axes[row][0].set_ylabel(name, rotation=0, labelpad=22, fontsize=8)
            for column, neighbor in enumerate(neighbors, start=1):
                axes[row][column].imshow(bank_images[neighbor].permute(1, 2, 0).numpy())
                color = "limegreen" if bank_labels[neighbor] == labels[query_id] else "red"
                for spine in axes[row][column].spines.values():
                    spine.set_edgecolor(color); spine.set_linewidth(3)
            for ax in axes[row]:
                ax.set_xticks([]); ax.set_yticks([])
            row += 1
    fig.savefig(save_path, dpi=130, bbox_inches="tight")


def main():
    datamodule = TinyImageNetDataModule()
    datamodule.setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = [name for name in ENCODERS if ENCODERS[name] is None or os.path.exists(ENCODERS[name])]

    if os.path.exists(FEATURE_CACHE):
        cached = torch.load(FEATURE_CACHE)
        train_features, val_features = cached["train"], cached["val"]
        train_labels, val_labels = cached["train_labels"], cached["val_labels"]
        names = list(train_features)
    else:
        train_features, val_features = {}, {}
        for name in names:
            extractor = load_extractor(name)
            train_features[name], train_labels = extract_features(extractor, datamodule.train_eval_dataloader(), device)
            val_features[name], val_labels = extract_features(extractor, datamodule.val_dataloader(), device)
        torch.save({"train": train_features, "val": val_features,
                    "train_labels": train_labels, "val_labels": val_labels}, FEATURE_CACHE)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("summaries", exist_ok=True)

    # 1. Which classes work, which fail (DAE vs random, same bank).
    per_class = {name: per_class_knn(train_features[name], train_labels, val_features[name], val_labels) for name in names}
    with open("summaries/partA_per_class_knn.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class_id"] + names)
        for class_id in range(200):
            writer.writerow([class_id] + [round(float(per_class[name][class_id]), 4) for name in names])
    reference = "dae" if "dae" in names else names[0]
    order = per_class[reference].argsort(descending=True)
    print(f"{reference} best 10 classes:", order[:10].tolist(), per_class[reference][order[:10]].tolist())
    print(f"{reference} worst 10 classes:", order[-10:].tolist(), per_class[reference][order[-10:]].tolist())
    print("classes at 0% k-NN:", {name: int((per_class[name] == 0).sum()) for name in names})

    # 2. Same 20 classes, t-SNE per encoder.
    plot_tsne(val_features, val_labels, "figures/a5_tsne.png")

    # 3. Nearest-neighbor retrieval grid.
    plot_retrieval(datamodule.val_ds.tensors[0], val_labels, datamodule.train_ds.tensors[0], train_labels,
                   val_features, train_features, "figures/a5_retrieval.png")

    # 4. Probe sanity: shuffled labels must land near chance (0.5%), else the probe leaks.
    for name in names:
        shuffled = train_labels[torch.randperm(len(train_labels), generator=torch.Generator().manual_seed(0))]
        probe = train_linear_probe(train_features[name], shuffled, num_classes=200, epochs=50, seed=42)
        print(f"{name}: probe on shuffled labels -> val top-1 =",
              round(top_k_accuracy(probe, val_features[name].float(), val_labels)["top1"], 4), "(chance 0.005)")


if __name__ == "__main__":
    main()
