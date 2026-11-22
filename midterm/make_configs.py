"""Write the config files of the experiment grids, in three stages that follow the results.

    python make_configs.py --stage search      # 30 random-search trials (needs nothing)
    python make_configs.py --stage seedcheck   # repeat the 3 best trials with 2 more seeds (needs search results)
    python make_configs.py --stage final       # pick the final model, write everything that depends on it

Each experiment is one YAML file (the assignment asks for a config per run). Only the keys that
differ from config.DEFAULTS are written. Nothing about the final model is typed in by hand: the
top trials come from the search results, the final model is the one with the best 3-seed mean
validation k-NN, and every later grid (evaluation, fine-tuning, transfer, robustness, test) is
written with that model's settings. run_plan.py calls the stages at the right moments.
"""
import argparse
import glob
import json
import os
import random
import statistics

import yaml

NUM_TRIALS = 30
SEARCH_SEED = 0  # separate from the training seed: fixes which trials are drawn
TOP_TRIALS = 3  # how many search trials get the multi-seed repeat
EXTRA_SEEDS = (43, 44)

# Strength means a different thing per corruption (sigma / patch drop rate / pixel hit rate),
# so each type gets its own sensible range. The mask range follows the plan (25-75% of patches).
STRENGTH_RANGE = {"gaussian": (0.1, 0.5), "mask": (0.25, 0.75), "salt_pepper": (0.05, 0.3)}
LATENT_DIMS = [64, 128, 256, 512]
LEARNING_RATES = [3e-4, 1e-3, 3e-3]
WEIGHT_DECAYS = [0.0, 1e-4]
HYPERPARAMETERS = ("corruption", "corruption_strength", "latent_dim", "learning_rate", "weight_decay")

# Proxy setting: a quarter of the train images and 20 epochs. Ranks trials at a fraction of the cost.
# k-NN is measured on the pooled conv layer, the layer chosen as h (feature_layer, see config.py).
PROXY = {"train_fraction": 0.25, "max_epochs": 20, "knn_every_n_epochs": 5, "feature_layer": "conv"}

# The final model trains until validation reconstruction error stops improving (by at least 1e-4 for
# 10 epochs) and keeps its best epoch. max_epochs is only a safety cap.
FULL_PRETRAIN = {"train_fraction": 1.0, "max_epochs": 100, "knn_every_n_epochs": 10, "feature_layer": "conv",
                 "early_stopping_patience": 10, "early_stopping_min_delta": 1e-4}

# Where the final pre-trained encoders are saved (train.py writes outputs/encoders/<run_name>.pt).
ENCODER_PATH = {"ae": "outputs/encoders/final_ae.pt", "dae": "outputs/encoders/final_dae.pt"}

# Fine-tuning has no fixed epoch count. Training stops when validation top-1 has not improved for
# `early_stopping_patience` epochs, and the best epoch's weights are kept. max_epochs is only a
# safety cap. A smaller label fraction means fewer steps per epoch (1% = 3 steps, 10% = 35,
# 100% = 351), so it gets a longer patience in epochs.
FINETUNE_STOP = {
    0.01: {"max_epochs": 1000, "early_stopping_patience": 100},
    0.1: {"max_epochs": 400, "early_stopping_patience": 25},
    1.0: {"max_epochs": 150, "early_stopping_patience": 8},
}

SELECTION_PATH = "configs/partA/final/selection.json"


def write_config(folder, run_name, settings):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{run_name}.yaml"), "w") as file:
        yaml.safe_dump({"run_name": run_name, **settings}, file, sort_keys=False)


# ---------------------------------------------------------------------------
# Stage 1: random search
# ---------------------------------------------------------------------------

def make_dae_search(folder="configs/partA/search"):
    """Random search over the DAE hyperparameters.

    Random search beats a grid at equal cost: with 30 trials each hyperparameter is
    tried at about 30 distinct values instead of 2-3, which matters when only a few
    of them (here probably corruption and latent_dim) drive the result.
    With coarse choices the same setting can be drawn twice, which would waste a run
    and make the top-3 repeat pick the same setting twice, so repeats are redrawn.
    """
    rng = random.Random(SEARCH_SEED)
    drawn = set()
    trial = 0
    while trial < NUM_TRIALS:
        corruption = rng.choice(list(STRENGTH_RANGE))
        low, high = STRENGTH_RANGE[corruption]
        settings = {
            "corruption": corruption,
            "corruption_strength": round(rng.uniform(low, high), 2),
            "latent_dim": rng.choice(LATENT_DIMS),
            "learning_rate": rng.choice(LEARNING_RATES),
            "weight_decay": rng.choice(WEIGHT_DECAYS),
        }
        key = tuple(settings.values())
        if key in drawn:
            continue
        drawn.add(key)
        write_config(folder, f"dae_search_{trial:02d}", {**PROXY, **settings})
        trial += 1


# ---------------------------------------------------------------------------
# Stage 2: repeat the best trials with more seeds
# ---------------------------------------------------------------------------

def read_result(run_name):
    with open(f"results/{run_name}.json") as file:
        return json.load(file)


def trial_settings(trial):
    with open(f"configs/partA/search/dae_search_{trial:02d}.yaml") as file:
        settings = yaml.safe_load(file)
    return {key: settings[key] for key in HYPERPARAMETERS}


def top_search_trials(count=TOP_TRIALS):
    """The `count` search trials with the highest validation k-NN (top-1 on the chosen layer h)."""
    scored = []
    for path in sorted(glob.glob("results/dae_search_*.json")):
        trial = int(os.path.basename(path)[len("dae_search_"):-len(".json")])
        with open(path) as file:
            scored.append((json.load(file)["metrics"]["val_knn_top1"], trial))
    if len(scored) < NUM_TRIALS:
        raise SystemExit(f"Only {len(scored)} of {NUM_TRIALS} search results found in results/. Run the search first.")
    return [trial for _, trial in sorted(scored, reverse=True)[:count]]


def make_seed_check(folder="configs/partA/seedcheck"):
    """Rerun the best search trials with new seeds.

    One seed on 10k validation images is noisy (about +-0.3 points), and the top trials
    differ by less than that. Choosing on the mean of 3 seeds avoids picking a lucky run.
    The original seed (42) is already in the search results.
    """
    for trial in top_search_trials():
        with open(f"configs/partA/search/dae_search_{trial:02d}.yaml") as file:
            settings = yaml.safe_load(file)
        settings.pop("run_name")
        for seed in EXTRA_SEEDS:
            write_config(folder, f"dae_seedcheck_{trial:02d}_s{seed}", {**settings, "seed": seed})


# ---------------------------------------------------------------------------
# Stage 3: choose the final model, write everything that depends on it
# ---------------------------------------------------------------------------

def choose_final_trial():
    """The trial with the best mean validation k-NN over its 3 seeds. Saves the evidence."""
    candidates = {}
    for trial in top_search_trials():
        scores = [read_result(f"dae_search_{trial:02d}")["metrics"]["val_knn_top1"]]
        for seed in EXTRA_SEEDS:
            scores.append(read_result(f"dae_seedcheck_{trial:02d}_s{seed}")["metrics"]["val_knn_top1"])
        candidates[trial] = scores
    best = max(candidates, key=lambda trial: statistics.mean(candidates[trial]))
    selection = {
        "chosen_trial": best, "chosen_settings": trial_settings(best),
        "candidates": {str(t): {"knn_top1_seeds_42_43_44": s, "mean": statistics.mean(s), "std": statistics.stdev(s),
                                "settings": trial_settings(t)} for t, s in candidates.items()},
    }
    os.makedirs(os.path.dirname(SELECTION_PATH), exist_ok=True)
    with open(SELECTION_PATH, "w") as file:
        json.dump(selection, file, indent=2)
    return selection["chosen_settings"]


def make_full_runs(chosen, folder="configs/partA/final"):
    """Full-size runs of the chosen DAE, its vanilla twin, and their evaluations.

    The vanilla AE is the same model with corruption strength 0, so the only
    difference to the DAE is the denoising objective. Evaluation runs are separate
    configs (mode: evaluate) and sort after the training runs.
    """
    write_config(folder, "final_dae", {**FULL_PRETRAIN, **chosen})
    write_config(folder, "final_ae", {**FULL_PRETRAIN, **chosen, "corruption_strength": 0.0})

    evaluation = {"mode": "evaluate", "probe_seeds": [42, 43, 44], "latent_dim": chosen["latent_dim"]}
    for model in ("dae", "ae"):
        for layer in ("bottleneck", "conv"):  # both layers: the which-layer-is-h comparison
            write_config(folder, f"full_eval_{model}_{layer}", {
                **evaluation, "feature_layer": layer, "encoder_path": ENCODER_PATH[model]})
    for layer in ("bottleneck", "conv"):  # random-init encoder: the "no learning" floor
        write_config(folder, f"full_eval_random_{layer}", {**evaluation, "feature_layer": layer})


def starting_points():
    return {"scratch": None, "ae": ENCODER_PATH["ae"], "dae": ENCODER_PATH["dae"]}


def finetune_settings(model, fraction, seed, latent_dim, dataset_settings=None):
    """One fine-tuning run on the pooled conv layer. Scratch moves everything at one rate; a pretrained
    encoder gets a gentler rate (1e-4) than the new head (1e-3)."""
    return {**(dataset_settings or {}), "mode": "finetune", "feature_layer": "conv", "latent_dim": latent_dim,
            "encoder_path": starting_points()[model], "train_fraction": fraction, **FINETUNE_STOP[fraction],
            "seed": seed, "learning_rate": 1e-3, "encoder_learning_rate": 1e-3 if model == "scratch" else 1e-4}


def make_finetune_runs(chosen, folder="configs/partA/finetune"):
    """Fine-tune each encoder (and a from-scratch baseline) at 1%, 10%, 100% of the labels, seed 42.

    scratch = same architecture, random init: the supervised baseline. All variants share one recipe;
    only the starting weights differ. There is no fixed epoch count: each run stops by the validation
    stop rule (FINETUNE_STOP) and keeps its best epoch. Seeds 43 and 44 are in make_finetune_seeds.
    """
    for model in starting_points():
        for fraction in FINETUNE_STOP:
            write_config(folder, f"ft_{model}_conv_{round(fraction * 100)}pct",
                         finetune_settings(model, fraction, 42, chosen["latent_dim"]))


def make_finetune_seeds(chosen, folder="configs/partA/finetune_extra"):
    """Seeds 43 and 44 for every fine-tuning run: the label-efficiency gaps are 1-3 points, so each run
    is repeated. The seed changes head init and batch order, not the class-balanced label subset."""
    for model in starting_points():
        for fraction in FINETUNE_STOP:
            for seed in EXTRA_SEEDS:
                write_config(folder, f"ft_{model}_conv_{round(fraction * 100)}pct_s{seed}",
                             finetune_settings(model, fraction, seed, chosen["latent_dim"]))


def make_transfer_runs(chosen, folder="configs/partA/transfer"):
    """Full protocol on the transfer dataset (CIFAR-100): the same encoders, nothing retrained.

    The encoders were pre-trained on Tiny-ImageNet only. Here they are (1) frozen and measured with
    k-NN and a linear probe, and (2) fine-tuned at 1/10/100% of the CIFAR labels next to a
    from-scratch baseline (pooled conv layer, 3 seeds, same recipe as the Tiny-ImageNet runs).
    """
    cifar = {"dataset": "cifar100"}
    for model in ("dae", "ae", "random"):
        for layer in ("bottleneck", "conv"):
            settings = {**cifar, "latent_dim": chosen["latent_dim"], "mode": "evaluate",
                        "probe_seeds": [42, 43, 44], "feature_layer": layer}
            if model != "random":
                settings["encoder_path"] = ENCODER_PATH[model]
            write_config(folder, f"cifar_eval_{model}_{layer}", settings)
    for model in starting_points():
        for fraction in FINETUNE_STOP:
            for seed in (42, 43, 44):
                write_config(folder, f"cifar_ft_{model}_conv_{round(fraction * 100)}pct_s{seed}",
                             finetune_settings(model, fraction, seed, chosen["latent_dim"], cifar))


def make_robustness_runs(chosen, folder="configs/partA/robustness"):
    """Probe accuracy under corrupted eval images, for the frozen encoders (pooled conv layer).

    Three encoders: DAE, vanilla AE, random-init. Supervised-from-scratch is left out because
    the fine-tuning runs do not save their weights and a probe on it is not the same protocol.
    """
    for model, encoder_path in {"dae": ENCODER_PATH["dae"], "ae": ENCODER_PATH["ae"], "random": None}.items():
        write_config(folder, f"robust_{model}_conv", {
            "mode": "robustness", "feature_layer": "conv", "encoder_path": encoder_path,
            "latent_dim": chosen["latent_dim"], "probe_seeds": [42, 43, 44]})


def make_test_runs(chosen, folder="configs/partA/test"):
    """The final numbers: the same recipes as the validation runs, scored on the test split.

    Nothing is tuned on these. The chosen DAE, its twin and the baselines are measured once: frozen
    k-NN + probe on both layers, and 3-seed fine-tuning on the pooled conv layer (the chosen h), for
    Tiny-ImageNet (official valid split) and CIFAR-100 (official test split). Fine-tuning retrains,
    since the validation runs do not save their weights.
    """
    for prefix, dataset_settings in (("test_", {}), ("test_cifar_", {"dataset": "cifar100"})):
        for model in ("dae", "ae", "random"):
            for layer in ("bottleneck", "conv"):
                settings = {**dataset_settings, "eval_split": "test", "latent_dim": chosen["latent_dim"],
                            "mode": "evaluate", "probe_seeds": [42, 43, 44], "feature_layer": layer}
                if model != "random":
                    settings["encoder_path"] = ENCODER_PATH[model]
                write_config(folder, f"{prefix}eval_{model}_{layer}", settings)
        for model in starting_points():
            for fraction in FINETUNE_STOP:
                for seed in (42, 43, 44):
                    write_config(folder, f"{prefix}ft_{model}_conv_{round(fraction * 100)}pct_s{seed}",
                                 {**finetune_settings(model, fraction, seed, chosen["latent_dim"], dataset_settings),
                                  "eval_split": "test"})


def make_final_stage():
    chosen = choose_final_trial()
    print(f"Chosen final model: {chosen}")
    make_full_runs(chosen)
    make_finetune_runs(chosen)
    make_finetune_seeds(chosen)
    make_transfer_runs(chosen)
    make_robustness_runs(chosen)
    make_test_runs(chosen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the experiment configs, stage by stage.")
    parser.add_argument("--stage", required=True, choices=["search", "seedcheck", "final"])
    stage = parser.parse_args().stage
    if stage == "search":
        make_dae_search()
        print(f"Wrote {NUM_TRIALS} search configs to configs/partA/search/")
    elif stage == "seedcheck":
        make_seed_check()
        print(f"Wrote seed-check configs for the top {TOP_TRIALS} trials to configs/partA/seedcheck/")
    else:
        make_final_stage()
        print("Wrote the final, finetune, finetune_extra, transfer, robustness and test configs.")
