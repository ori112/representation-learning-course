"""Export the training curves of finished runs from TensorBoard logs, and plot them.

    uv run python -m scripts.export_curves --logs outputs/tb_logs --out summaries/curves.csv
    uv run python -m scripts.export_curves --plot summaries/curves.csv figures/curves.png

Question it answers: was every run trained long enough? A curve that is still falling at
the last epoch means the number of epochs set in the config cut training short, and a
validation curve that turns upward means it ran too long.
"""
import argparse
import csv
import fnmatch
import glob
import os

SCALAR_TAGS = ("train_loss", "val_recon_loss", "val_knn_top1", "val_top1", "val_top5", "epoch")
DEFAULT_RUNS = ("full_*", "ft_*_conv_*pct", "ft_*_conv_*pct_s4*", "dae_search_*", "tune_*")


def export(logs_dir, out_path, run_patterns):
    """Write one long CSV (run, tag, step, value) with the scalar curves of the matching runs."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    rows = []
    for run_dir in sorted(glob.glob(os.path.join(logs_dir, "*"))):
        run_name = os.path.basename(run_dir)
        if not any(fnmatch.fnmatch(run_name, pattern) for pattern in run_patterns):
            continue
        # Lightning writes outputs/tb_logs/<run>/version_N/events.*; a resumed run may have several versions.
        for version_dir in sorted(glob.glob(os.path.join(run_dir, "version_*"))):
            accumulator = EventAccumulator(version_dir, size_guidance={"scalars": 0})
            accumulator.Reload()
            for tag in accumulator.Tags()["scalars"]:
                if tag in SCALAR_TAGS:
                    rows += [[run_name, os.path.basename(version_dir), tag, event.step, event.value]
                             for event in accumulator.Scalars(tag)]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run", "version", "tag", "step", "value"])
        writer.writerows(rows)
    print(f"wrote {out_path}: {len(rows)} points from {len({row[0] for row in rows})} runs")


def last_epoch_slope(csv_path, tag="train_loss", fraction=0.2):
    """For every run: how much the curve still fell over the last `fraction` of its points.

    Prints start, end and the relative drop over the last stretch. A drop of several percent
    in the last 20% of training means the run was stopped while still improving.
    """
    series = {}
    with open(csv_path) as file:
        for row in csv.DictReader(file):
            if row["tag"] == tag:
                series.setdefault(row["run"], []).append((int(row["step"]), float(row["value"])))
    for run, points in sorted(series.items()):
        values = [value for _, value in sorted(points)]
        tail = values[int(len(values) * (1 - fraction)):] or values[-2:]
        if len(tail) >= 2 and tail[0] > 0:
            print(f"{run:34s} start {values[0]:8.4f}  end {values[-1]:8.4f}  last-{int(fraction * 100)}% drop {100 * (tail[0] - tail[-1]) / tail[0]:5.1f}%")


def plot(csv_path, save_path, tags=("train_loss", "val_top1", "val_knn_top1", "val_recon_loss")):
    """One panel per tag, one line per run."""
    import matplotlib.pyplot as plt

    series = {}
    with open(csv_path) as file:
        for row in csv.DictReader(file):
            series.setdefault((row["tag"], row["run"]), []).append((int(row["step"]), float(row["value"])))
    fig, axes = plt.subplots(1, len(tags), figsize=(4.6 * len(tags), 3.8))
    for ax, tag in zip(axes, tags):
        for (series_tag, run), points in sorted(series.items()):
            if series_tag == tag:
                points.sort()
                ax.plot([p[0] for p in points], [p[1] for p in points], label=run, linewidth=1)
        ax.set_title(tag)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export or plot TensorBoard scalar curves.")
    parser.add_argument("--logs", default="outputs/tb_logs")
    parser.add_argument("--out", default="summaries/curves.csv")
    parser.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS), help="run-name glob patterns")
    parser.add_argument("--plot", nargs=2, metavar=("CSV", "PNG"))
    parser.add_argument("--tail", metavar="CSV", help="print the how-much-was-it-still-falling table")
    args = parser.parse_args()
    if args.plot:
        plot(*args.plot)
    elif args.tail:
        last_epoch_slope(args.tail)
    else:
        export(args.logs, args.out, args.runs)
