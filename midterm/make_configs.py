"""Write the config files of the experiment grids, in stages that follow the results.

    python make_configs.py --stage search      # 30 random-search trials (needs nothing)
    python make_configs.py --stage seedcheck   # repeat the 3 best trials with 2 more seeds (needs search results)
    python make_configs.py --stage final       # pick the final model; write pre-training, frozen evaluation, robustness
    python make_configs.py --stage tune        # fine-tuning learning-rate grid at 10% labels (needs the final encoders)
    python make_configs.py --stage finetune    # pick each start's learning rate; write fine-tuning, transfer, test

Each experiment is one YAML file (the assignment asks for a config per run). Only the keys that
differ from config.DEFAULTS are written. Nothing about the final model or the fine-tuning recipe is
typed in by hand: every choice is read from the validation results of the previous stage.
run_plan.py calls the stages at the right moments.
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
SEEDS = (42,) + EXTRA_SEEDS  # every model is trained, and every protocol repeated, on these seeds

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

# The final models train until validation reconstruction error stops improving (by at least 1e-4 for
# 10 epochs) and keep their best epoch. max_epochs is only a safety cap.
FULL_PRETRAIN = {"train_fraction": 1.0, "max_epochs": 100, "knn_every_n_epochs": 10, "feature_layer": "conv",
                 "early_stopping_patience": 10, "early_stopping_min_delta": 1e-4}

# Fine-tuning has no fixed epoch count. Training stops when validation top-1 has not improved for
# `early_stopping_patience` epochs, and the best epoch's weights are kept. max_epochs is only a
# safety cap. A smaller label fraction means fewer steps per epoch (1% = 3 steps, 10% = 35,
# 100% = 351), so it gets a longer patience in epochs.
FINETUNE_STOP = {
    0.01: {"max_epochs": 1000, "early_stopping_patience": 100},
    0.1: {"max_epochs": 400, "early_stopping_patience": 25},
    1.0: {"max_epochs": 150, "early_stopping_patience": 8},
}

# Fine-tuning learning-rate grid, at 10% labels, validation only. For a from-scratch model one rate moves
# every weight. For a pretrained start the new head keeps 1e-3 and the encoder rate is the one tuned.
TUNE_RATES = {"scratch": (3e-4, 1e-3, 3e-3), "ae": (1e-4, 3e-4, 1e-3), "dae": (1e-4, 3e-4, 1e-3)}
TUNE_WEIGHT_DECAYS = (0.0, 1e-4)

SELECTION_PATH = "configs/partA/final/selection.json"
TUNING_PATH = "configs/partA/tune/selection.json"


def encoder_path(model, seed):
    """Where train.py saves the encoder of final_<model>[_s<seed>] (outputs/encoders/<run_name>.pt)."""
    return f"outputs/encoders/final_{model}{'' if seed == 42 else f'_s{seed}'}.pt"


def suffix(seed):
    """Run-name suffix of a repeat: seed 42 has none, the others _s43, _s44."""
    return "" if seed == 42 else f"_s{seed}"


def write_config(folder, run_name, settings):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{run_name}.yaml"), "w") as file:
        yaml.safe_dump({"run_name": run_name, **settings}, file, sort_keys=False)


def read_result(run_name):
    with open(f"results/{run_name}.json") as file:
        return json.load(file)


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
# Stage 3: choose the final model; write pre-training, frozen evaluation, robustness
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


def evaluation_settings(model, layer, seed, latent_dim, extra=None):
    """A frozen evaluation (mode: evaluate) of one encoder. The random-init encoder depends on the seed too."""
    settings = {**(extra or {}), "mode": "evaluate", "seed": seed, "probe_seeds": [42, 43, 44],
                "latent_dim": latent_dim, "feature_layer": layer}
    if model != "random":
        settings["encoder_path"] = encoder_path(model, seed)
    return settings


def make_full_runs(chosen, folder="configs/partA/final"):
    """Full-size runs of the chosen DAE and its vanilla twin on 3 seeds, and their frozen evaluations.

    The vanilla AE is the same model with corruption strength 0, so the only difference to the DAE is
    the denoising objective. Each pre-training seed gives its own encoder, so every later number can be
    reported as a mean and std over pre-training seeds, not just over probes. The random-init
    baseline also takes one random encoder per seed.
    """
    for seed in SEEDS:
        write_config(folder, f"final_dae{suffix(seed)}", {**FULL_PRETRAIN, **chosen, "seed": seed})
        write_config(folder, f"final_ae{suffix(seed)}", {**FULL_PRETRAIN, **chosen, "corruption_strength": 0.0, "seed": seed})
        for model in ("dae", "ae", "random"):
            for layer in ("bottleneck", "conv"):  # both layers: the which-layer-is-h comparison
                write_config(folder, f"full_eval_{model}_{layer}{suffix(seed)}",
                             evaluation_settings(model, layer, seed, chosen["latent_dim"]))


def make_robustness_runs(chosen, folder="configs/partA/robustness"):
    """Probe accuracy under corrupted eval images, for the frozen encoders (pooled conv layer), 3 seeds.

    Three encoders: DAE, vanilla AE, random-init. Supervised-from-scratch is left out because
    the fine-tuning runs do not save their weights and a probe on it is not the same protocol.
    """
    for seed in SEEDS:
        for model in ("dae", "ae", "random"):
            settings = {"mode": "robustness", "seed": seed, "feature_layer": "conv",
                        "latent_dim": chosen["latent_dim"], "probe_seeds": [42, 43, 44]}
            if model != "random":
                settings["encoder_path"] = encoder_path(model, seed)
            write_config(folder, f"robust_{model}_conv{suffix(seed)}", settings)


def make_final_stage():
    chosen = choose_final_trial()
    print(f"Chosen final model: {chosen}")
    make_full_runs(chosen)
    make_robustness_runs(chosen)


# ---------------------------------------------------------------------------
# Stage 4: tune the fine-tuning learning rates (10% labels, validation only)
# ---------------------------------------------------------------------------

def latent_dim_of_final():
    with open(SELECTION_PATH) as file:
        return json.load(file)["chosen_settings"]["latent_dim"]


def rate_settings(start, rate, weight_decay):
    """Scratch moves everything at one rate. A pretrained start keeps the head at 1e-3 and tunes the encoder rate."""
    return {"learning_rate": rate if start == "scratch" else 1e-3, "encoder_learning_rate": rate,
            "weight_decay": weight_decay}


def make_tuning_runs(folder="configs/partA/tune"):
    """Learning rate and weight decay grid at 10% labels, for scratch, vanilla AE and DAE (seed 42, validation).

    Every start gets the same size grid, so neither the supervised baseline nor the pretrained models
    can be called undertuned. The winner per start is used for every label fraction and both datasets.
    """
    latent_dim = latent_dim_of_final()
    for start, rates in TUNE_RATES.items():
        for rate in rates:
            for weight_decay in TUNE_WEIGHT_DECAYS:
                write_config(folder, f"tune_{start}_lr{rate:g}_wd{weight_decay:g}", {
                    "mode": "finetune", "feature_layer": "conv", "latent_dim": latent_dim, "seed": 42,
                    "encoder_path": None if start == "scratch" else encoder_path(start, 42),
                    "train_fraction": 0.1, **FINETUNE_STOP[0.1], **rate_settings(start, rate, weight_decay)})


# ---------------------------------------------------------------------------
# Stage 5: pick the fine-tuning recipe per start; write fine-tuning, transfer and test configs
# ---------------------------------------------------------------------------

def choose_finetune_recipes():
    """For each start, the (rate, weight decay) with the best validation top-1 in the tuning grid."""
    recipes, table = {}, {}
    for start, rates in TUNE_RATES.items():
        scored = []
        for rate in rates:
            for weight_decay in TUNE_WEIGHT_DECAYS:
                top1 = read_result(f"tune_{start}_lr{rate:g}_wd{weight_decay:g}")["metrics"]["val_top1"]
                scored.append((top1, rate, weight_decay))
        table[start] = [{"rate": r, "weight_decay": w, "val_top1": t} for t, r, w in scored]
        _, rate, weight_decay = max(scored)
        recipes[start] = rate_settings(start, rate, weight_decay)
    with open(TUNING_PATH, "w") as file:
        json.dump({"chosen": recipes, "grid": table}, file, indent=2)
    return recipes


def finetune_settings(start, fraction, seed, latent_dim, recipes, dataset_settings=None):
    """One fine-tuning run on the pooled conv layer. Seed s starts from the encoder pre-trained with seed s."""
    return {**(dataset_settings or {}), "mode": "finetune", "feature_layer": "conv", "latent_dim": latent_dim,
            "encoder_path": None if start == "scratch" else encoder_path(start, seed),
            "train_fraction": fraction, **FINETUNE_STOP[fraction], "seed": seed, **recipes[start]}


def make_finetune_runs(latent_dim, recipes, folder_42="configs/partA/finetune", folder_extra="configs/partA/finetune_extra"):
    """Fine-tune each encoder (and a from-scratch baseline) at 1%, 10%, 100% of the labels, 3 seeds.

    scratch = same architecture, random init: the supervised baseline. All variants share the same
    recipe, except the learning rates chosen for each start in the tuning stage; only the starting
    weights differ. The seed changes head init, batch order and, for pretrained starts, which
    pre-trained encoder is the start. The class-balanced label subset is identical across seeds.
    """
    for start in recipes:
        for fraction in FINETUNE_STOP:
            for seed in SEEDS:
                write_config(folder_42 if seed == 42 else folder_extra, f"ft_{start}_conv_{round(fraction * 100)}pct{suffix(seed)}",
                             finetune_settings(start, fraction, seed, latent_dim, recipes))


def make_transfer_runs(latent_dim, recipes, folder="configs/partA/transfer"):
    """Full protocol on the transfer dataset (CIFAR-100): the same encoders, nothing retrained.

    The encoders were pre-trained on Tiny-ImageNet only. Here they are (1) frozen and measured with
    k-NN and a linear probe, and (2) fine-tuned at 1/10/100% of the CIFAR labels next to a
    from-scratch baseline (pooled conv layer, 3 seeds, same recipe as the Tiny-ImageNet runs).
    """
    cifar = {"dataset": "cifar100"}
    for seed in SEEDS:
        for model in ("dae", "ae", "random"):
            for layer in ("bottleneck", "conv"):
                write_config(folder, f"cifar_eval_{model}_{layer}{suffix(seed)}",
                             evaluation_settings(model, layer, seed, latent_dim, cifar))
    for start in recipes:
        for fraction in FINETUNE_STOP:
            for seed in SEEDS:
                write_config(folder, f"cifar_ft_{start}_conv_{round(fraction * 100)}pct_s{seed}",
                             finetune_settings(start, fraction, seed, latent_dim, recipes, cifar))


def make_test_runs(latent_dim, recipes, folder="configs/partA/test"):
    """The final numbers: the same recipes as the validation runs, scored on the test split.

    Nothing is tuned on these. Frozen k-NN + probe on both layers, and 3-seed fine-tuning on the
    pooled conv layer (the chosen h), for Tiny-ImageNet (official valid split) and CIFAR-100
    (official test split). Fine-tuning retrains, since the validation runs do not save their weights.
    """
    for prefix, dataset_settings in (("test_", {}), ("test_cifar_", {"dataset": "cifar100"})):
        for seed in SEEDS:
            for model in ("dae", "ae", "random"):
                for layer in ("bottleneck", "conv"):
                    write_config(folder, f"{prefix}eval_{model}_{layer}{suffix(seed)}",
                                 evaluation_settings(model, layer, seed, latent_dim,
                                                     {**dataset_settings, "eval_split": "test"}))
        for start in recipes:
            for fraction in FINETUNE_STOP:
                for seed in SEEDS:
                    write_config(folder, f"{prefix}ft_{start}_conv_{round(fraction * 100)}pct_s{seed}",
                                 {**finetune_settings(start, fraction, seed, latent_dim, recipes, dataset_settings),
                                  "eval_split": "test"})


def make_finetune_stage():
    recipes = choose_finetune_recipes()
    print(f"Chosen fine-tuning recipes: {recipes}")
    latent_dim = latent_dim_of_final()
    make_finetune_runs(latent_dim, recipes)
    make_transfer_runs(latent_dim, recipes)
    make_test_runs(latent_dim, recipes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the experiment configs, stage by stage.")
    parser.add_argument("--stage", required=True, choices=["search", "seedcheck", "final", "tune", "finetune"])
    stage = parser.parse_args().stage
    if stage == "search":
        make_dae_search()
        print(f"Wrote {NUM_TRIALS} search configs to configs/partA/search/")
    elif stage == "seedcheck":
        make_seed_check()
        print(f"Wrote seed-check configs for the top {TOP_TRIALS} trials to configs/partA/seedcheck/")
    elif stage == "final":
        make_final_stage()
        print("Wrote the pre-training, frozen evaluation and robustness configs.")
    elif stage == "tune":
        make_tuning_runs()
        print("Wrote the fine-tuning tuning grid to configs/partA/tune/")
    else:
        make_finetune_stage()
        print("Wrote the finetune, finetune_extra, transfer and test configs.")
