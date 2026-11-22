"""Run the whole Part A experiment plan in order, one config per run through train.py.

    python run_plan.py              # everything, in order. Finished runs are skipped.
    python run_plan.py --dry-run    # list what would run and what would be moved aside
    python run_plan.py --only test  # one step: pretrain | frozen | robustness | finetune | seeds | transfer | test

After a Colab disconnect, run it again: runs whose results/<run_name>.json exists are skipped,
and an interrupted run resumes from its last checkpoint. A failing run stops the plan, so a
problem is seen right away and not buried under later runs.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

# (step name, config glob). Order matters: later steps need the encoders saved by "pretrain".
PLAN = [
    ("pretrain", "configs/partA/final/final_*.yaml"),
    ("frozen", "configs/partA/final/full_eval_*.yaml"),
    ("robustness", "configs/partA/robustness/*.yaml"),
    ("finetune", "configs/partA/finetune/*_conv_*.yaml"),   # the bottleneck fine-tuning runs are not used in the report
    ("seeds", "configs/partA/finetune_extra/ft_*.yaml"),
    ("transfer", "configs/partA/transfer/*.yaml"),
    ("test", "configs/partA/test/*.yaml"),                  # the final numbers, run once at the end
]

# Result files of these runs were written before the validation stop rule existed. They are moved
# aside (not deleted) so the same configs run again. A result is old if its saved config has no
# early_stopping_patience key. Search, seed-check and the first pre-training runs are kept.
STALE_PATTERNS = ["full_eval_*", "ft_*", "tune_*", "cifar_eval_*", "cifar_ft_*", "robust_*", "test_*"]
ARCHIVE_DIR = "results/old_fixed_epochs"


def move_stale_results(dry_run):
    """Move old result files aside. Returns the run names that were (or would be) moved."""
    moved_names = set()
    moved = 0
    for pattern in STALE_PATTERNS:
        for path in sorted(glob.glob(f"results/{pattern}.json")):
            with open(path) as file:
                if "early_stopping_patience" in json.load(file)["config"]:
                    continue
            moved += 1
            moved_names.add(os.path.basename(path)[:-5])
            if dry_run:
                print(f"  would move {path}")
            else:
                os.makedirs(ARCHIVE_DIR, exist_ok=True)
                shutil.move(path, ARCHIVE_DIR)
    print(f"{'would move' if dry_run else 'moved'} {moved} old result files to {ARCHIVE_DIR}/")
    return moved_names


def main():
    parser = argparse.ArgumentParser(description="Run the Part A experiment plan.")
    parser.add_argument("--only", choices=[name for name, _ in PLAN], help="run just this step")
    parser.add_argument("--dry-run", action="store_true", help="show what would happen, run nothing")
    args = parser.parse_args()

    moved_names = move_stale_results(args.dry_run)  # in a dry run these still exist on disk, count them as gone
    for step, config_glob in PLAN:
        if args.only and step != args.only:
            continue
        configs = sorted(glob.glob(config_glob))
        todo = [path for path in configs if os.path.basename(path)[:-5] in moved_names
                or not os.path.exists(f"results/{os.path.basename(path)[:-5]}.json")]
        print(f"\n=== {step}: {len(todo)} of {len(configs)} runs to do ({config_glob})")
        for index, config_path in enumerate(todo, start=1):
            print(f"--- {step} {index}/{len(todo)}: {config_path}", flush=True)
            if not args.dry_run:
                subprocess.run([sys.executable, "train.py", "--config", config_path], check=True)
    print("\nPlan finished." if not args.dry_run else "\nDry run only, nothing was run.")


if __name__ == "__main__":
    main()
