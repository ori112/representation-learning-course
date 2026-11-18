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
    full = {"train_fraction": 1.0, "max_epochs": 100, "knn_every_n_epochs": 10}
    write_config(folder, "full_dae", {**full, **chosen})
    write_config(folder, "full_ae", {**full, **chosen, "corruption_strength": 0.0})

    evaluation = {"mode": "evaluate", "probe_seeds": [42, 43, 44], "latent_dim": chosen["latent_dim"]}
    for model in ("dae", "ae"):
        for layer in ("bottleneck", "conv"):  # conv = which-layer-is-h comparison
            write_config(folder, f"full_eval_{model}_{layer}", {
                **evaluation, "feature_layer": layer, "encoder_path": f"outputs/encoders/full_{model}.pt"})
    for layer in ("bottleneck", "conv"):  # random-init encoder: the "no learning" floor
        write_config(folder, f"full_eval_random_{layer}", {**evaluation, "feature_layer": layer})


if __name__ == "__main__":
    make_dae_search()
    make_full_runs()
    make_seed_check()
    print(f"Wrote {NUM_TRIALS} search configs and the seed-check configs under configs/partA/")
