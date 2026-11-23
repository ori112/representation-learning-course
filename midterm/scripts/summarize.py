"""Aggregate result JSONs into the small CSV tables in summaries/.

    uv run python -m scripts.summarize                                   # results/ -> summaries/
    uv run python -m scripts.summarize --results final_results --out summaries

Every number in the report's tables and plots comes from these CSVs, and every CSV comes from
a result JSON written by train.py, so the chain config -> JSON -> CSV -> figure can be rerun.
All accuracies are in percent. (per_class_knn and robustness_feature_shift are written by
scripts/interpret.py and scripts/scale_check.py.)
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics

STARTS = ("scratch", "ae", "dae")
DATASETS = {"tiny_imagenet": "", "cifar100": "cifar_"}   # run-name prefix of the transfer runs


def load_runs(results_dir):
    """{run_name: {config, metrics, ...}} for every result JSON in the folder."""
    runs = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as file:
            runs[os.path.basename(path)[:-5]] = json.load(file)
    return runs


def write_csv(out_dir, name, header, rows):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {name}: {len(rows)} rows")


def percent(value, digits=2):
    return round(100 * value, digits)


def search_tables(runs, out_dir):
    """Random-search trials and the multi-seed repeat of the best ones (validation, proxy setting)."""
    rows = []
    for name in sorted(runs):
        match = re.fullmatch(r"dae_search_(\d+)", name)
        if match:
            c, m = runs[name]["config"], runs[name]["metrics"]
            rows.append([int(match[1]), c["corruption"], c["corruption_strength"], c["latent_dim"], c["learning_rate"],
                         c["weight_decay"], round(m["val_recon_loss"], 5), percent(m["val_knn_top1"]),
                         percent(m["val_knn_top5"]), percent(m["val_knn_top1_bottleneck"]), percent(m["val_knn_top1_conv"])])
    write_csv(out_dir, "dae_search_summary.csv",
              ["trial", "corruption", "corruption_strength", "latent_dim", "learning_rate", "weight_decay",
               "val_recon_loss", "val_knn_top1_h", "val_knn_top5_h", "val_knn_top1_bottleneck", "val_knn_top1_conv"], rows)

    repeated = sorted({int(re.fullmatch(r"dae_seedcheck_(\d+)_s\d+", n)[1])
                       for n in runs if re.fullmatch(r"dae_seedcheck_(\d+)_s\d+", n)})
    rows = []
    for trial in repeated:
        names = [(42, f"dae_search_{trial:02d}")] + [(int(n.rsplit("_s", 1)[1]), n) for n in sorted(runs)
                                                     if n.startswith(f"dae_seedcheck_{trial:02d}_s")]
        for seed, name in names:
            m = runs[name]["metrics"]
            rows.append([trial, seed, round(m["val_recon_loss"], 5), percent(m["val_knn_top1"]),
                         percent(m["val_knn_top1_bottleneck"]), percent(m["val_knn_top1_conv"])])
    write_csv(out_dir, "dae_seedcheck_summary.csv",
              ["trial", "seed", "val_recon_loss", "val_knn_top1_h", "val_knn_top1_bottleneck", "val_knn_top1_conv"], rows)


def seed_suffix(seed):
    return "" if seed == 42 else f"_s{seed}"


def mean_std(values):
    """Mean and sample std (n-1) of a list. The std is 0 for a single value."""
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def pretrain_table(runs, out_dir):
    """The final pre-training runs (3 seeds each): how long they trained and where they ended."""
    rows = []
    for model in ("dae", "ae"):
        for seed in (42, 43, 44):
            run = runs.get(f"final_{model}{seed_suffix(seed)}")
            if run:
                m, c = run["metrics"], run["config"]
                rows.append([model, seed, c["corruption"], c["corruption_strength"], c["latent_dim"], m["epochs_trained"],
                             m["best_epoch"], round(m["val_recon_loss"], 5), percent(m["val_knn_top1_bottleneck"]),
                             percent(m["val_knn_top1_conv"]), run["train_seconds"]])
    write_csv(out_dir, "pretrain_summary.csv",
              ["model", "seed", "corruption", "corruption_strength", "latent_dim", "epochs_trained", "best_epoch",
               "val_recon_loss", "val_knn_top1_bottleneck", "val_knn_top1_conv", "train_seconds"], rows)


def frozen_table(runs, out_dir, split):
    """k-NN and linear probe of frozen encoders: mean and std over the 3 pre-training seeds (each with 3 probes)."""
    rows = []
    for dataset, dataset_prefix in DATASETS.items():
        for model in ("random", "ae", "dae"):
            for layer in ("bottleneck", "conv"):
                found = []
                for seed in (42, 43, 44):
                    base = f"{dataset_prefix}eval_{model}_{layer}{seed_suffix(seed)}"
                    name = ("full_" + base if not dataset_prefix else base) if split == "val" else "test_" + base
                    if name in runs:
                        found.append(runs[name]["metrics"])
                if not found:
                    continue
                stats = {key: mean_std([100 * m[key] for m in found])
                         for key in ("knn_top1", "knn_top5", "probe_top1_mean", "probe_top5_mean")}
                rows.append([dataset, model, layer, int(found[0]["feature_dim"]),
                             round(stats["knn_top1"][0], 2), round(stats["knn_top1"][1], 2), round(stats["knn_top5"][0], 2),
                             round(stats["probe_top1_mean"][0], 2), round(stats["probe_top1_mean"][1], 2),
                             round(stats["probe_top5_mean"][0], 2), len(found)])
    write_csv(out_dir, f"frozen_{split}_summary.csv",
              ["dataset", "model", "layer", "feature_dim", "knn_top1_mean", "knn_top1_std", "knn_top5_mean",
               "probe_top1_mean", "probe_top1_std", "probe_top5_mean", "n_seeds"], rows)


def finetune_table(runs, out_dir, split):
    """Fine-tuning at 1/10/100% labels: mean and sample std (n-1) over the 3 seeds, pooled conv layer."""
    rows = []
    for dataset, dataset_prefix in DATASETS.items():
        for start in STARTS:
            for pct in (1, 10, 100):
                top1, top5, epochs = [], [], []
                for seed in (42, 43, 44):
                    if split == "val":
                        # seed 42 of the Tiny-ImageNet runs has no seed suffix, the others do
                        name = (f"cifar_ft_{start}_conv_{pct}pct_s{seed}" if dataset_prefix
                                else f"ft_{start}_conv_{pct}pct" + ("" if seed == 42 else f"_s{seed}"))
                    else:
                        name = f"test_{dataset_prefix}ft_{start}_conv_{pct}pct_s{seed}"
                    if name in runs:
                        m = runs[name]["metrics"]
                        top1.append(100 * m[f"{split}_top1"])
                        top5.append(100 * m[f"{split}_top5"])
                        epochs.append(m["epochs_trained"])
                if top1:
                    spread = statistics.stdev if len(top1) > 1 else (lambda values: 0.0)
                    rows.append([dataset, start, pct, round(statistics.mean(top1), 2), round(spread(top1), 2),
                                 round(statistics.mean(top5), 2), round(spread(top5), 2),
                                 round(statistics.mean(epochs), 1), len(top1)])
    write_csv(out_dir, f"finetune_{split}_summary.csv",
              ["dataset", "start", "label_pct", "top1_mean", "top1_std", "top5_mean", "top5_std",
               "epochs_trained_mean", "n_seeds"], rows)


def robustness_table(runs, out_dir):
    """Probe and k-NN accuracy per corruption and severity (0 = clean), validation split.
    Mean and sample std over the 3 pre-training seeds."""
    rows = []
    for model in ("ae", "dae", "random"):
        found = [runs[f"robust_{model}_conv{seed_suffix(seed)}"] for seed in (42, 43, 44)
                 if f"robust_{model}_conv{seed_suffix(seed)}" in runs]
        if not found:
            continue
        for kind in found[0]["severities"]:
            for severity in range(0, 6):
                keys = ("clean_probe_top1", "clean_knn_top1") if severity == 0 else                     (f"{kind}_s{severity}_probe_top1", f"{kind}_s{severity}_knn_top1")
                probe = mean_std([100 * run["metrics"][keys[0]] for run in found])
                knn = mean_std([100 * run["metrics"][keys[1]] for run in found])
                rows.append([model, kind, severity, round(probe[0], 3), round(probe[1], 3), round(knn[0], 3),
                             round(knn[1], 3), len(found)])
    write_csv(out_dir, "robustness_val_summary.csv",
              ["model", "corruption", "severity", "probe_top1_mean", "probe_top1_std", "knn_top1_mean", "knn_top1_std",
               "n_seeds"], rows)


def tuning_table(runs, out_dir):
    """The fine-tuning learning-rate grid at 10% labels (validation, seed 42) and which setting won."""
    rows = []
    for name in sorted(runs):
        match = re.fullmatch(r"tune_(scratch|ae|dae)_lr[\d.e-]+_wd[\d.e-]+", name)
        if match:
            c = runs[name]["config"]
            rows.append([match[1], c["encoder_learning_rate"], c["learning_rate"], c["weight_decay"],
                         percent(runs[name]["metrics"]["val_top1"])])
    rows.sort(key=lambda row: (row[0], row[1], row[3]))
    write_csv(out_dir, "finetune_tuning_summary.csv",
              ["start", "encoder_learning_rate", "head_learning_rate", "weight_decay", "val_top1"], rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate result JSONs into summaries/*.csv")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="summaries")
    args = parser.parse_args()
    runs = load_runs(args.results)
    print(f"read {len(runs)} result files from {args.results}/")
    search_tables(runs, args.out)
    pretrain_table(runs, args.out)
    for split in ("val", "test"):
        frozen_table(runs, args.out, split)
        finetune_table(runs, args.out, split)
    robustness_table(runs, args.out)
    tuning_table(runs, args.out)


if __name__ == "__main__":
    main()
