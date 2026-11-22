"""Aggregate results/*.json into the small CSV tables in summaries/.

    uv run python -m scripts.summarize                 # reads results/, writes summaries/
    uv run python -m scripts.summarize --out /tmp/chk  # write elsewhere, to diff against summaries/

Every number in the report's tables and plots comes from these CSVs, and every CSV comes from
a result JSON written by train.py, so the chain config -> JSON -> CSV -> figure can be rerun.
(per_class_knn, robustness_feature_shift are written by scripts/interpret.py and scale_check.py.)
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics

LAYER_NAME = {"conv": "conv_pooled"}  # name used in the Tiny-ImageNet table


def load_metrics(results_dir):
    """{run_name: metrics dict}, plus the config of each run."""
    runs = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as file:
            content = json.load(file)
        runs[os.path.basename(path)[:-5]] = content
    return runs


def write_csv(out_dir, name, header, rows):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {name}: {len(rows)} rows")


def search_tables(runs, out_dir):
    """Random-search trials, and the top trials repeated with extra seeds."""
    rows = []
    for name in sorted(runs):
        match = re.fullmatch(r"dae_search_(\d+)", name)
        if match:
            config, metrics = runs[name]["config"], runs[name]["metrics"]
            rows.append([int(match[1]), config["corruption"], f"{config['corruption_strength']:.2f}", config["latent_dim"],
                         config["learning_rate"], config["weight_decay"], f"{metrics['val_recon_loss']:.5f}",
                         f"{metrics['val_knn_top1']:.4f}", f"{metrics['val_knn_top5']:.4f}"])
    write_csv(out_dir, "dae_search_summary.csv",
              ["trial", "corruption", "corruption_strength", "latent_dim", "learning_rate", "weight_decay",
               "val_recon_loss", "val_knn_top1", "val_knn_top5"], rows)

    seed_rows = []
    for name in sorted(runs):
        match = re.fullmatch(r"dae_seedcheck_(\d+)_s(\d+)", name)
        if match:
            metrics = runs[name]["metrics"]
            seed_rows.append([int(match[1]), int(match[2]), f"{metrics['val_recon_loss']:.5f}",
                              f"{metrics['val_knn_top1']:.4f}", f"{metrics['val_knn_top5']:.4f}"])
            # the original seed (42) of this trial is the search run itself
    for trial in sorted({row[0] for row in seed_rows}):
        metrics = runs[f"dae_search_{trial:02d}"]["metrics"]
        seed_rows.append([trial, 42, f"{metrics['val_recon_loss']:.5f}",
                          f"{metrics['val_knn_top1']:.4f}", f"{metrics['val_knn_top5']:.4f}"])
    trial_order = {2: 0, 15: 1, 8: 2}  # the order of make_seed_check: best first
    write_csv(out_dir, "dae_seedcheck_summary.csv", ["trial", "seed", "val_recon_loss", "val_knn_top1", "val_knn_top5"],
              sorted(seed_rows, key=lambda row: (trial_order.get(row[0], row[0]), row[1])))


def frozen_tables(runs, out_dir):
    """k-NN and probe on frozen encoders: Tiny-ImageNet (full_eval_*) and CIFAR-100 (cifar_eval_*)."""
    rows = []
    for model in ("random", "ae", "dae"):
        for layer in ("bottleneck", "conv"):
            name = f"full_eval_{model}_{layer}"
            if name in runs:
                m = runs[name]["metrics"]
                rows.append([model, LAYER_NAME.get(layer, layer), int(m["feature_dim"]), f"{m['knn_top1']:.4f}",
                             f"{m['knn_top5']:.4f}", f"{m['probe_top1_mean']:.4f}", f"{m['probe_top1_std']:.4f}",
                             f"{m['probe_top5_mean']:.4f}"])
    write_csv(out_dir, "partA_full_val_summary.csv",
              ["model", "layer", "feature_dim", "knn_top1", "knn_top5", "probe_top1_mean", "probe_top1_std",
               "probe_top5_mean"], rows)

    rows = []
    for model in ("dae", "ae", "random"):
        for layer in ("bottleneck", "conv"):
            name = f"cifar_eval_{model}_{layer}"
            if name in runs:
                m = runs[name]["metrics"]
                rows.append([model, layer] + [round(100 * m[key], 2) for key in
                                              ("knn_top1", "knn_top5", "probe_top1_mean", "probe_top5_mean")])
    write_csv(out_dir, "cifar100_frozen_val_summary.csv",
              ["model", "layer", "knn_top1", "knn_top5", "probe_top1_mean", "probe_top5_mean"], rows)


def finetune_tables(runs, out_dir):
    """Fine-tuning: every Tiny-ImageNet run, the 3-seed means, the 10% tuning grid, and CIFAR-100."""
    rows = []
    for start in ("scratch", "ae", "dae"):
        for layer in ("bottleneck", "conv"):
            for pct in (1, 10, 100):
                name = f"ft_{start}_{layer}_{pct}pct"
                if name in runs:
                    m = runs[name]["metrics"]
                    rows.append([start, layer, f"{pct / 100:.2f}", f"{m['val_top1']:.4f}", f"{m['val_top5']:.4f}",
                                 f"{m['train_loss']:.5f}"])
    write_csv(out_dir, "partA_finetune_val_summary.csv",
              ["start", "layer", "label_fraction", "val_top1", "val_top5", "final_train_loss"], rows)

    def three_seed(prefix, file_name, seed42_suffix=""):
        """Mean and sample std (n-1) over seeds 42/43/44 of the conv runs, in percent."""
        table = []
        for start in ("scratch", "ae", "dae"):
            for pct in (1, 10, 100):
                values = []
                for seed in (42, 43, 44):
                    suffix = seed42_suffix if seed == 42 else f"_s{seed}"
                    name = f"{prefix}{start}_conv_{pct}pct{suffix}"
                    if name in runs:
                        values.append(100 * runs[name]["metrics"]["val_top1"])
                if values:
                    table.append([start, pct, round(statistics.mean(values), 2),
                                  round(statistics.stdev(values), 2), len(values)])
        write_csv(out_dir, file_name, ["start", "label_pct", "val_top1_mean", "val_top1_std", "n_seeds"], table)

    three_seed("ft_", "partA_finetune_3seed_summary.csv")                       # seed 42 file has no suffix
    three_seed("cifar_ft_", "cifar100_finetune_3seed_summary.csv", "_s42")      # CIFAR seed 42 file is _s42

    tuning = []
    for name in sorted(runs):
        match = re.fullmatch(r"tune_(scratch|dae)_lr([\d.e-]+)_wd([\d.e-]+)", name)
        if match:
            tuning.append([match[1], float(match[2]), float(match[3]), round(100 * runs[name]["metrics"]["val_top1"], 2)])
    for start, name in (("scratch", "ft_scratch_conv_10pct"), ("dae", "ft_dae_conv_10pct")):
        if name in runs:  # the default recipe: scratch lr 1e-3, DAE encoder lr 1e-4, weight decay 0
            tuning.append([start, 1e-3 if start == "scratch" else 1e-4, 0.0, round(100 * runs[name]["metrics"]["val_top1"], 2)])
    tuning.sort(key=lambda row: (row[0] != "scratch", -row[1], row[2]))
    write_csv(out_dir, "partA_finetune_tuning_10pct.csv", ["start", "encoder_lr", "weight_decay", "val_top1_pct_seed42"],
              tuning)


def robustness_table(runs, out_dir):
    """Probe and k-NN accuracy per corruption and severity (0 = clean)."""
    rows = []
    for model in ("ae", "dae", "random"):
        name = f"robust_{model}_conv"
        if name not in runs:
            continue
        m = runs[name]["metrics"]
        for kind in runs[name]["severities"]:
            rows.append([model, kind, 0, m["clean_probe_top1"], m["clean_knn_top1"]])
            for severity in range(1, 6):
                rows.append([model, kind, severity, m[f"{kind}_s{severity}_probe_top1"],
                             m[f"{kind}_s{severity}_knn_top1"]])
    write_csv(out_dir, "robustness_val_summary.csv", ["model", "corruption", "severity", "probe_top1", "knn_top1"], rows)


def test_tables(runs, out_dir):
    """Final numbers on the test split (test_* runs), in percent: frozen and fine-tuned, both datasets."""
    datasets = {"tiny_imagenet": "test_", "cifar100": "test_cifar_"}
    frozen = []
    for dataset, prefix in datasets.items():
        for model in ("random", "ae", "dae"):
            for layer in ("bottleneck", "conv"):
                name = f"{prefix}eval_{model}_{layer}"
                if name in runs:
                    m = runs[name]["metrics"]
                    frozen.append([dataset, model, layer] + [round(100 * m[key], 2) for key in
                                   ("knn_top1", "knn_top5", "probe_top1_mean", "probe_top1_std", "probe_top5_mean")])
    write_csv(out_dir, "partA_test_frozen_summary.csv",
              ["dataset", "model", "layer", "knn_top1", "knn_top5", "probe_top1_mean", "probe_top1_std",
               "probe_top5_mean"], frozen)

    finetuned = []
    for dataset, prefix in datasets.items():
        for start in ("scratch", "ae", "dae"):
            for pct in (1, 10, 100):
                top1, top5 = [], []
                for seed in (42, 43, 44):
                    name = f"{prefix}ft_{start}_conv_{pct}pct_s{seed}"
                    if name in runs:
                        top1.append(100 * runs[name]["metrics"]["test_top1"])
                        top5.append(100 * runs[name]["metrics"]["test_top5"])
                if top1:
                    spread = statistics.stdev if len(top1) > 1 else (lambda values: 0.0)
                    finetuned.append([dataset, start, pct, round(statistics.mean(top1), 2), round(spread(top1), 2),
                                      round(statistics.mean(top5), 2), round(spread(top5), 2), len(top1)])
    write_csv(out_dir, "partA_test_finetune_summary.csv",
              ["dataset", "start", "label_pct", "test_top1_mean", "test_top1_std", "test_top5_mean", "test_top5_std",
               "n_seeds"], finetuned)


def main():
    parser = argparse.ArgumentParser(description="Aggregate results/*.json into summaries/*.csv")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="summaries")
    args = parser.parse_args()
    runs = load_metrics(args.results)
    print(f"read {len(runs)} result files from {args.results}/")
    search_tables(runs, args.out)
    frozen_tables(runs, args.out)
    finetune_tables(runs, args.out)
    robustness_table(runs, args.out)
    test_tables(runs, args.out)


if __name__ == "__main__":
    main()
