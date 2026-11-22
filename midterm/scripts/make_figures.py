"""Draw every figure of the report from the CSVs in summaries/ (run scripts.summarize first).

    uv run python -m scripts.make_figures

Figures that need a trained encoder (t-SNE, retrieval grid, per-class k-NN, feature shift) come from
scripts/interpret.py and scripts/scale_check.py instead.
"""
import os

from scripts.analysis import plot_label_efficiency, plot_layer_choice, plot_recon_vs_knn, plot_robustness

SUMMARIES = "summaries"
FIGURES = "figures"


def main():
    os.makedirs(FIGURES, exist_ok=True)
    jobs = [
        ("a2_recon_vs_knn.png", "dae_search_summary.csv", plot_recon_vs_knn, {}),
        ("a2_layer_choice.png", "dae_search_summary.csv", plot_layer_choice, {}),
        ("a4_label_efficiency_val.png", "finetune_val_summary.csv", plot_label_efficiency, {"title_suffix": " (val)"}),
        ("a4_label_efficiency_test.png", "finetune_test_summary.csv", plot_label_efficiency, {"title_suffix": " (test)"}),
        ("a4_robustness.png", "robustness_val_summary.csv", plot_robustness, {}),
    ]
    for figure_name, csv_name, plot, options in jobs:
        csv_path = os.path.join(SUMMARIES, csv_name)
        if not os.path.exists(csv_path) or sum(1 for _ in open(csv_path)) < 2:
            print(f"skip {figure_name}: {csv_name} is missing or empty")
            continue
        plot(csv_path, os.path.join(FIGURES, figure_name), **options)
        print(f"wrote figures/{figure_name}")


if __name__ == "__main__":
    main()
