"""Plot functions for the report. Each one reads a CSV written by scripts/summarize.py."""
import csv

import matplotlib.pyplot as plt

CORRUPTION_COLORS = {"gaussian": "tab:blue", "mask": "tab:orange", "salt_pepper": "tab:green"}
START_COLORS = {"scratch": "tab:gray", "ae": "tab:green", "dae": "tab:red"}
START_LABELS = {"scratch": "supervised from scratch", "ae": "vanilla AE init", "dae": "DAE init"}


def read_rows(csv_path):
    with open(csv_path) as file:
        return list(csv.DictReader(file))


def save(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_recon_vs_knn(search_csv, save_path=None):
    """Validation reconstruction error vs validation k-NN accuracy, one dot per search trial.

    The required A2 figure. If reconstruction were a good proxy for representation quality the dots
    would fall on a falling line (lower error, higher accuracy). A flat or scattered cloud is the
    evidence for tuning by k-NN instead. The k-NN is measured on h, the pooled conv layer.
    """
    rows = read_rows(search_csv)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for corruption, color in CORRUPTION_COLORS.items():
        subset = [row for row in rows if row["corruption"] == corruption]
        ax.scatter([float(r["val_recon_loss"]) for r in subset], [float(r["val_knn_top1_h"]) for r in subset],
                   c=color, label=corruption, s=40)
    ax.set_xlabel("validation reconstruction error (MSE, clean input)")
    ax.set_ylabel("validation k-NN top-1 on h (%)")
    ax.set_title("Reconstruction error vs representation quality")
    ax.legend()
    ax.grid(alpha=0.3)
    return save(fig, save_path)


def plot_layer_choice(search_csv, save_path=None):
    """k-NN on the bottleneck against k-NN on the pooled conv layer, one dot per search trial.

    Supports the choice of h: dots above the diagonal mean the conv layer is the better representation
    for that trial. Every one of the 30 trials gives one comparison.
    """
    rows = read_rows(search_csv)
    fig, ax = plt.subplots(figsize=(4.6, 4.5))
    bottleneck = [float(r["val_knn_top1_bottleneck"]) for r in rows]
    conv = [float(r["val_knn_top1_conv"]) for r in rows]
    ax.scatter(bottleneck, conv, s=36, c="tab:purple")
    low, high = min(bottleneck + conv), max(bottleneck + conv)
    ax.plot([low, high], [low, high], color="gray", linestyle="--", label="equal")
    ax.set_xlabel("k-NN top-1 on the 512-d bottleneck (%)")
    ax.set_ylabel("k-NN top-1 on the pooled conv map (%)")
    ax.set_title("Which layer is h? (validation, 30 trials)")
    ax.legend()
    ax.grid(alpha=0.3)
    return save(fig, save_path)


def plot_label_efficiency(finetune_csv, save_path=None, title_suffix=""):
    """Fine-tuning accuracy vs label fraction, one panel per dataset (required A4 figure).

    Error bars are the sample std over 3 seeds. The x axis is logarithmic because the fractions are
    1%, 10%, 100%. The supervised-from-scratch line is the reference: a pretrained start is useful
    where its curve sits above it.
    """
    rows = read_rows(finetune_csv)
    datasets = [("tiny_imagenet", "Tiny-ImageNet (200 classes)"), ("cifar100", "CIFAR-100 transfer (100 classes)")]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.2))
    for ax, (dataset, title) in zip(axes, datasets):
        for start, color in START_COLORS.items():
            points = sorted((float(r["label_pct"]), float(r["top1_mean"]), float(r["top1_std"]))
                            for r in rows if r["dataset"] == dataset and r["start"] == start)
            ax.errorbar([p[0] for p in points], [p[1] for p in points], yerr=[p[2] for p in points],
                        marker="o", capsize=3, color=color, label=START_LABELS[start])
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1%", "10%", "100%"])
        ax.set_xlabel("fraction of labels")
        ax.set_ylabel(f"top-1 accuracy (%){title_suffix}")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    return save(fig, save_path)


def plot_robustness(summary_csv, save_path=None):
    """Accuracy vs corruption severity, one column per corruption, one row per metric.

    Severity 0 is the clean image. The probe and k-NN rows are kept apart on purpose: they can react
    very differently to the same damage. Lines are the mean over 3 pre-training seeds, bands one std.
    """
    colors = {"dae": "tab:red", "ae": "tab:green", "random": "tab:gray"}
    labels = {"dae": "DAE", "ae": "vanilla AE", "random": "random init"}
    rows = read_rows(summary_csv)
    corruptions = list(dict.fromkeys(row["corruption"] for row in rows))
    fig, axes = plt.subplots(2, len(corruptions), figsize=(3.2 * len(corruptions), 5.6), sharey="row")
    for row_index, (metric, title) in enumerate((("probe_top1", "linear probe"), ("knn_top1", "k-NN"))):
        for ax, corruption in zip(axes[row_index], corruptions):
            for model, color in colors.items():
                points = sorted((int(r["severity"]), float(r[f"{metric}_mean"]), float(r[f"{metric}_std"]))
                                for r in rows if r["model"] == model and r["corruption"] == corruption)
                x = [p[0] for p in points]
                mean = [p[1] for p in points]
                std = [p[2] for p in points]
                ax.plot(x, mean, marker="o", color=color, label=labels[model])
                ax.fill_between(x, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                                color=color, alpha=0.2)
            ax.set_xticks(range(6))
            ax.grid(alpha=0.3)
            if row_index == 0:
                ax.set_title(corruption.replace("_", " "))
            if row_index == 1:
                ax.set_xlabel("severity (0 = clean)")
        axes[row_index][0].set_ylabel(f"{title} top-1 (%)")
    axes[0][0].legend()
    fig.tight_layout()
    return save(fig, save_path)
