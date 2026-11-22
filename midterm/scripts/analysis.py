"""Helpers for the analysis notebooks: load result JSONs, draw the required plots."""
import glob
import json

import matplotlib.pyplot as plt

CORRUPTION_COLORS = {"gaussian": "tab:blue", "mask": "tab:orange", "salt_pepper": "tab:green"}


def load_results(pattern):
    """Read every results/<run>.json matching the glob into a list of flat dicts.

    Config keys and metric keys are merged (they never overlap), so a row reads
    row["latent_dim"], row["val_knn_top1"], and so on.
    """
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as file:
            result = json.load(file)
        rows.append({**result["config"], **result["metrics"], "train_seconds": result.get("train_seconds")})
    return rows


def plot_recon_vs_knn(rows, save_path=None):
    """Validation reconstruction error vs validation k-NN accuracy, one dot per trial.

    The required A2 figure. If reconstruction were a good proxy for representation
    quality the dots would fall on a falling line (lower error, higher accuracy).
    A flat or scattered cloud is the evidence for tuning by k-NN instead.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for corruption, color in CORRUPTION_COLORS.items():
        subset = [row for row in rows if row["corruption"] == corruption]
        ax.scatter([row["val_recon_loss"] for row in subset],
                   [100 * row["val_knn_top1"] for row in subset],
                   c=color, label=corruption, s=40)
    ax.set_xlabel("validation reconstruction error (MSE, clean input)")
    ax.set_ylabel("validation k-NN top-1 (%)")
    ax.set_title("Reconstruction error vs representation quality")
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_label_efficiency(summary_csvs, save_path=None):
    """Fine-tuning accuracy vs label fraction, one panel per dataset (required A4 figure).

    summary_csvs maps a panel title to a 3-seed summary CSV (start, label_pct, mean, std).
    Error bars are the std over 3 seeds. The x axis is logarithmic because the fractions
    are 1%, 10%, 100%. The supervised-from-scratch line is the reference: a pretrained
    start is useful where its curve sits above it.
    """
    import csv

    colors = {"scratch": "tab:gray", "ae": "tab:green", "dae": "tab:red"}
    labels = {"scratch": "supervised from scratch", "ae": "vanilla AE init", "dae": "DAE init"}
    fig, axes = plt.subplots(1, len(summary_csvs), figsize=(5.2 * len(summary_csvs), 4.2), sharey=False)
    for ax, (title, csv_path) in zip(axes, summary_csvs.items()):
        with open(csv_path) as file:
            rows = list(csv.DictReader(file))
        for start, color in colors.items():
            points = sorted((float(row["label_pct"]), float(row["val_top1_mean"]), float(row["val_top1_std"]))
                            for row in rows if row["start"] == start)
            ax.errorbar([p[0] for p in points], [p[1] for p in points], yerr=[p[2] for p in points],
                        marker="o", capsize=3, color=color, label=labels[start])
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1%", "10%", "100%"])
        ax.set_xlabel("fraction of labels")
        ax.set_ylabel("validation top-1 (%)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_robustness(summary_csv, save_path=None):
    """Accuracy vs corruption severity, one column per corruption, one row per metric.

    Severity 0 is the clean image. The probe and k-NN rows are kept apart on purpose:
    they react very differently to the same damage (see the robustness discussion).
    """
    import csv

    colors = {"dae": "tab:red", "ae": "tab:green", "random": "tab:gray"}
    labels = {"dae": "DAE", "ae": "vanilla AE", "random": "random init"}
    with open(summary_csv) as file:
        rows = list(csv.DictReader(file))
    corruptions = list(dict.fromkeys(row["corruption"] for row in rows))
    fig, axes = plt.subplots(2, len(corruptions), figsize=(3.2 * len(corruptions), 5.6), sharey="row")
    for row_index, (metric, title) in enumerate((("probe_top1", "linear probe"), ("knn_top1", "k-NN"))):
        for ax, corruption in zip(axes[row_index], corruptions):
            for model, color in colors.items():
                points = sorted((int(r["severity"]), 100 * float(r[metric]))
                                for r in rows if r["model"] == model and r["corruption"] == corruption)
                ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", color=color, label=labels[model])
            ax.set_xticks(range(6))
            ax.grid(alpha=0.3)
            if row_index == 0:
                ax.set_title(corruption.replace("_", " "))
            if row_index == 1:
                ax.set_xlabel("severity (0 = clean)")
        axes[row_index][0].set_ylabel(f"{title} top-1 (%)")
    axes[0][0].legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
