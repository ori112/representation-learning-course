"""Write the config files of the experiment grids: python make_configs.py

Each experiment is one YAML file (the assignment asks for a config per run), so the
search is reproducible and every trial can be rerun alone. Only the keys that differ
from config.DEFAULTS are written.
"""
import os
import random
import yaml

NUM_TRIALS = 30
SEARCH_SEED = 0  # separate from the training seed: fixes which trials are drawn

# Strength means a different thing per corruption (sigma / patch drop rate / pixel hit rate),
# so each type gets its own sensible range. The mask range follows the plan (25-75% of patches).
STRENGTH_RANGE = {"gaussian": (0.1, 0.5), "mask": (0.25, 0.75), "salt_pepper": (0.05, 0.3)}
LATENT_DIMS = [64, 128, 256, 512]
LEARNING_RATES = [3e-4, 1e-3, 3e-3]
WEIGHT_DECAYS = [0.0, 1e-4]

# Proxy setting: a quarter of the train images and 20 epochs. Ranks trials at a fraction of the cost.
PROXY = {"train_fraction": 0.25, "max_epochs": 20, "knn_every_n_epochs": 5}

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


def write_config(folder, run_name, settings):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{run_name}.yaml"), "w") as file:
        yaml.safe_dump({"run_name": run_name, **settings}, file, sort_keys=False)


def make_dae_search(folder="configs/partA/search"):
    """Random search over the DAE hyperparameters.

    Random search beats a grid at equal cost: with 30 trials each hyperparameter is
    tried at about 30 distinct values instead of 2-3, which matters when only a few
    of them (here probably corruption and latent_dim) drive the result.
    """
    rng = random.Random(SEARCH_SEED)
    # Note: with coarse choices, a config can be drawn twice. Trial 10 repeats trial 02,
    # and the run reproduced trial 02's numbers exactly, so only 29 trials are distinct.
    for trial in range(NUM_TRIALS):
        corruption = rng.choice(list(STRENGTH_RANGE))
        low, high = STRENGTH_RANGE[corruption]
        write_config(folder, f"dae_search_{trial:02d}", {
            **PROXY,
            "corruption": corruption,
            "corruption_strength": round(rng.uniform(low, high), 2),
            "latent_dim": rng.choice(LATENT_DIMS),
            "learning_rate": rng.choice(LEARNING_RATES),
            "weight_decay": rng.choice(WEIGHT_DECAYS),
        })


def make_seed_check(trials=(2, 15, 8), extra_seeds=(43, 44), folder="configs/partA/seedcheck"):
    """Rerun the best search trials with new seeds.

    One seed on 10k validation images is noisy (about +-0.25 points), and the top trials
    differ by less than that. Choosing on the mean of 3 seeds avoids picking a lucky run.
    The original seed (42) is already in the search results.
    """
    for trial in trials:
        with open(f"configs/partA/search/dae_search_{trial:02d}.yaml") as file:
            settings = yaml.safe_load(file)
        settings.pop("run_name")
        for seed in extra_seeds:
            write_config(folder, f"dae_seedcheck_{trial:02d}_s{seed}", {**settings, "seed": seed})


def make_full_runs(folder="configs/partA/final"):
    """Full-size runs of the chosen DAE, its vanilla twin, and their evaluations.

    The vanilla AE is the same model with corruption strength 0, so the only
    difference to the DAE is the denoising objective. Evaluation runs are separate
    configs (mode: evaluate) and sort after the training runs, so a loop over the
    folder trains first and evaluates second.
    """
    chosen = {"corruption": "mask", "corruption_strength": 0.30, "latent_dim": 512,
              "learning_rate": 1e-3, "weight_decay": 0.0}
    # Full size, same stop rule for both models: stop when validation reconstruction error has not
    # improved by at least 1e-4 for 10 epochs, keep the best epoch. max_epochs is only a safety cap.
    # (The first versions of these runs, full_dae / full_ae, trained a fixed 100 epochs.)
    full = {"train_fraction": 1.0, "max_epochs": 100, "knn_every_n_epochs": 10,
            "early_stopping_patience": 10, "early_stopping_min_delta": 1e-4}
    write_config(folder, "final_dae", {**full, **chosen})
    write_config(folder, "final_ae", {**full, **chosen, "corruption_strength": 0.0})

    evaluation = {"mode": "evaluate", "probe_seeds": [42, 43, 44], "latent_dim": chosen["latent_dim"]}
    for model in ("dae", "ae"):
        for layer in ("bottleneck", "conv"):  # conv = which-layer-is-h comparison
            write_config(folder, f"full_eval_{model}_{layer}", {
                **evaluation, "feature_layer": layer, "encoder_path": ENCODER_PATH[model]})
    for layer in ("bottleneck", "conv"):  # random-init encoder: the "no learning" floor
        write_config(folder, f"full_eval_random_{layer}", {**evaluation, "feature_layer": layer})


def make_finetune_runs(folder="configs/partA/finetune"):
    """Fine-tune each encoder (and a from-scratch baseline) at 1%, 10%, 100% of the labels.

    scratch = same architecture, random init: the supervised baseline. All variants share
    one recipe; only the starting weights differ. There is no fixed epoch count: each run
    stops by the validation stop rule (FINETUNE_STOP) and keeps its best epoch.
    Pretrained encoders get a gentler encoder learning rate (1e-4) than the new head (1e-3).
    """
    starting_points = {"scratch": None, "ae": ENCODER_PATH["ae"], "dae": ENCODER_PATH["dae"]}
    for model, encoder_path in starting_points.items():
        for layer in ("bottleneck", "conv"):
            for fraction in FINETUNE_STOP:
                write_config(folder, f"ft_{model}_{layer}_{round(fraction * 100)}pct", {
                    "mode": "finetune", "feature_layer": layer, "encoder_path": encoder_path,
                    "latent_dim": 512, "train_fraction": fraction, **FINETUNE_STOP[fraction],
                    "learning_rate": 1e-3,
                    "encoder_learning_rate": 1e-3 if model == "scratch" else 1e-4,
                })


def make_finetune_extras(folder="configs/partA/finetune_extra"):
    """Two checks on the fine-tuning results, both on the pooled conv layer.

    1. Seeds: the label-efficiency gaps (1-3 points) are small, so every conv run is
       repeated with 2 more seeds. The seed changes head init and batch order, not the
       class-balanced label subset, which stays identical across models and seeds.
    2. Baseline tuning: at 10% labels, scratch and DAE get a small grid over learning rate
       and weight decay (validation only), so neither side can be called undertuned.
       The default combination was already run in make_finetune_runs, so it is skipped.
    """
    starting_points = {"scratch": None, "ae": ENCODER_PATH["ae"], "dae": ENCODER_PATH["dae"]}
    base = {"mode": "finetune", "feature_layer": "conv", "latent_dim": 512}

    for model, encoder_path in starting_points.items():
        for fraction in FINETUNE_STOP:
            for seed in (43, 44):
                write_config(folder, f"ft_{model}_conv_{round(fraction * 100)}pct_s{seed}", {
                    **base, "encoder_path": encoder_path, "train_fraction": fraction, **FINETUNE_STOP[fraction],
                    "seed": seed, "learning_rate": 1e-3,
                    "encoder_learning_rate": 1e-3 if model == "scratch" else 1e-4})

    # scratch moves everything at one rate. For a pretrained encoder the head rate stays 1e-3
    # and the encoder rate is the one tuned.
    for model, encoder_path, rates, default_rate in (
            ("scratch", None, (3e-4, 1e-3, 3e-3), 1e-3), ("dae", ENCODER_PATH["dae"], (1e-4, 3e-4, 1e-3), 1e-4)):
        for rate in rates:
            for weight_decay in (0.0, 1e-4, 1e-3):
                if rate == default_rate and weight_decay == 0.0:
                    continue
                write_config(folder, f"tune_{model}_lr{rate:g}_wd{weight_decay:g}", {
                    **base, "encoder_path": encoder_path, "train_fraction": 0.1, **FINETUNE_STOP[0.1],
                    "learning_rate": rate if model == "scratch" else 1e-3,
                    "encoder_learning_rate": rate, "weight_decay": weight_decay})


def make_transfer_runs(folder="configs/partA/transfer"):
    """Full protocol on the transfer dataset (CIFAR-100): the same encoders, nothing retrained.

    The encoders were pre-trained on Tiny-ImageNet only. Here they are (1) frozen and measured
    with k-NN and a linear probe, and (2) fine-tuned at 1/10/100% of the CIFAR labels next to
    a from-scratch baseline. Fine-tuning uses the pooled conv layer (the chosen h) and 3 seeds,
    with the same recipe as the Tiny-ImageNet runs. Eval configs sort before the ft_ ones.
    """
    base = {"dataset": "cifar100", "latent_dim": 512}
    starting_points = {"scratch": None, "ae": ENCODER_PATH["ae"], "dae": ENCODER_PATH["dae"]}

    for model in ("dae", "ae", "random"):
        for layer in ("bottleneck", "conv"):
            settings = {**base, "mode": "evaluate", "probe_seeds": [42, 43, 44], "feature_layer": layer}
            if model != "random":
                settings["encoder_path"] = starting_points[model]
            write_config(folder, f"cifar_eval_{model}_{layer}", settings)

    for model, encoder_path in starting_points.items():
        for fraction in FINETUNE_STOP:
            for seed in (42, 43, 44):
                write_config(folder, f"cifar_ft_{model}_conv_{round(fraction * 100)}pct_s{seed}", {
                    **base, "mode": "finetune", "feature_layer": "conv", "encoder_path": encoder_path,
                    "train_fraction": fraction, **FINETUNE_STOP[fraction], "seed": seed, "learning_rate": 1e-3,
                    "encoder_learning_rate": 1e-3 if model == "scratch" else 1e-4})


def make_robustness_runs(folder="configs/partA/robustness"):
    """Probe accuracy under corrupted eval images, for the frozen encoders (pooled conv layer).

    Three encoders: DAE, vanilla AE, random-init. Supervised-from-scratch is left out because
    the fine-tuning runs do not save their weights and a probe on it is not the same protocol.
    """
    starting_points = {"dae": ENCODER_PATH["dae"], "ae": ENCODER_PATH["ae"], "random": None}
    for model, encoder_path in starting_points.items():
        write_config(folder, f"robust_{model}_conv", {
            "mode": "robustness", "feature_layer": "conv", "encoder_path": encoder_path,
            "latent_dim": 512, "probe_seeds": [42, 43, 44]})


def make_test_runs(folder="configs/partA/test"):
    """The final numbers: the same recipes as the validation runs, scored on the test split.

    Nothing was tuned on these. The chosen DAE, its twin and the baselines are measured once:
    frozen k-NN + probe on both layers, and 3-seed fine-tuning on the pooled conv layer (the
    chosen h). Both datasets, Tiny-ImageNet (official valid split) and CIFAR-100 (official test split).
    Fine-tuning has to retrain, since the validation runs did not save their weights.
    """
    starting_points = {"scratch": None, "ae": ENCODER_PATH["ae"], "dae": ENCODER_PATH["dae"]}
    for prefix, dataset_settings in (("test_", {}), ("test_cifar_", {"dataset": "cifar100"})):
        base = {**dataset_settings, "eval_split": "test", "latent_dim": 512}
        for model in ("dae", "ae", "random"):
            for layer in ("bottleneck", "conv"):
                settings = {**base, "mode": "evaluate", "probe_seeds": [42, 43, 44], "feature_layer": layer}
                if model != "random":
                    settings["encoder_path"] = starting_points[model]
                write_config(folder, f"{prefix}eval_{model}_{layer}", settings)
        for model, encoder_path in starting_points.items():
            for fraction in FINETUNE_STOP:
                for seed in (42, 43, 44):
                    write_config(folder, f"{prefix}ft_{model}_conv_{round(fraction * 100)}pct_s{seed}", {
                        **base, "mode": "finetune", "feature_layer": "conv", "encoder_path": encoder_path,
                        "train_fraction": fraction, **FINETUNE_STOP[fraction], "seed": seed, "learning_rate": 1e-3,
                        "encoder_learning_rate": 1e-3 if model == "scratch" else 1e-4})


if __name__ == "__main__":
    make_dae_search()
    make_finetune_extras()
    make_finetune_runs()
    make_full_runs()
    make_seed_check()
    make_transfer_runs()
    make_robustness_runs()
    make_test_runs()
    print(f"Wrote {NUM_TRIALS} search configs and the other grids under configs/partA/")
