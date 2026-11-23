"""Run the whole Part A experiment plan from zero, in order, one config per run through train.py.

    python run_plan.py              # everything except the test set. Finished runs are skipped.
    python run_plan.py --only test  # the final test-set runs (do this once, after the rest looks right)
    python run_plan.py --dry-run    # list what would run, run nothing

The order is the experiment itself:
  search      30 random-search trials of the denoising autoencoder (proxy setting)
  seedcheck   the 3 best trials again with 2 more seeds (picked automatically from the search results)
  pretrain    the final DAE and its vanilla twin on 3 seeds, from the trial with the best 3-seed mean (automatic)
  frozen, robustness   k-NN, probe and corruption tests of those encoders (3 seeds each, random-init baseline too)
  tune        fine-tuning learning-rate grid at 10% labels for scratch, vanilla AE and DAE, and the best per start is picked
  finetune, seeds, transfer   fine-tuning at 1/10/100% labels (3 seeds) on Tiny-ImageNet and on CIFAR-100
  test        the same evaluations on the test split, once

Nothing about the final model is typed in: make_configs.py reads the results and writes the next
configs ("stages"). After a Colab disconnect, run it again: runs whose results/<run_name>.json exists
are skipped and an interrupted run resumes from its last checkpoint. A failing run stops the plan,
so a problem is seen right away and not buried under later runs.
"""
import argparse
import glob
import os
import subprocess
import sys

# (step name, config glob, make_configs stage to run first or None, only when asked for)
PLAN = [
    ("search", "configs/partA/search/*.yaml", "search", False),
    ("seedcheck", "configs/partA/seedcheck/*.yaml", "seedcheck", False),
    ("pretrain", "configs/partA/final/final_*.yaml", "final", False),
    ("frozen", "configs/partA/final/full_eval_*.yaml", None, False),
    ("robustness", "configs/partA/robustness/*.yaml", None, False),
    ("tune", "configs/partA/tune/*.yaml", "tune", False),
    ("finetune", "configs/partA/finetune/*.yaml", "finetune", False),
    ("seeds", "configs/partA/finetune_extra/*.yaml", None, False),
    ("transfer", "configs/partA/transfer/*.yaml", None, False),
    ("test", "configs/partA/test/*.yaml", "finetune", True),  # the final numbers: run once, at the end
]


def run(command, dry_run):
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the Part A experiment plan.")
    parser.add_argument("--only", choices=[step[0] for step in PLAN], help="run just this step")
    parser.add_argument("--dry-run", action="store_true", help="show what would happen, run nothing")
    args = parser.parse_args()

    for step, config_glob, stage, only_when_asked in PLAN:
        if (args.only and step != args.only) or (only_when_asked and args.only != step):
            continue
        if stage is not None:
            run([sys.executable, "make_configs.py", "--stage", stage], args.dry_run)
        configs = sorted(glob.glob(config_glob))
        todo = [path for path in configs if not os.path.exists(f"results/{os.path.basename(path)[:-5]}.json")]
        print(f"\n=== {step}: {len(todo)} of {len(configs)} runs to do ({config_glob})", flush=True)
        for index, config_path in enumerate(todo, start=1):
            print(f"--- {step} {index}/{len(todo)}: {config_path}", flush=True)
            run([sys.executable, "train.py", "--config", config_path], args.dry_run)
    print("\nDry run only, nothing was run." if args.dry_run else "\nPlan finished.")


if __name__ == "__main__":
    main()
