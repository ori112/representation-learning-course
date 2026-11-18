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


if __name__ == "__main__":
    make_dae_search()
    print(f"Wrote {NUM_TRIALS} configs to configs/partA/search/")
