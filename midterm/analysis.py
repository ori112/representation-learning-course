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
