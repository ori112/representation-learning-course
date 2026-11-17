"""Run configuration: defaults here, overridden by one YAML file per experiment.

Every run is `python train.py --config configs/<name>.yaml`. The YAML only lists
what differs from DEFAULTS, so each file reads as "what this experiment changes".
"""
import yaml

DEFAULTS = {
    # --- run identity and paths
    "run_name": "dae_default",
    "mode": "pretrain_dae",        # which model train.py builds (see utils.build_model)
    "seed": 42,                    # fixed seed: same init and data order on every rerun
    "data_dir": "dataset/",        # cached Tiny-ImageNet tensors live here
    "output_dir": "outputs",       # checkpoints, TensorBoard logs, saved encoders (put on Drive in Colab)
    "results_dir": "results",      # small metric JSONs, committed to git

    # --- data
    "batch_size": 256,
    "train_fraction": 1.0,         # balanced fraction of the 90k train images (0.25 = proxy run)
    "num_workers": 0,              # data is in memory as tensors, workers would only add overhead

    # --- training
    "max_epochs": 100,
    "precision": "auto",           # "16-mixed" on GPU, "32-true" on CPU
    "resume": True,                # continue from outputs/checkpoints/<run_name>/last.ckpt if present
    "fast_dev_run": False,         # one batch through every step: a wiring test, not a real run

    # --- model: denoising autoencoder
    "latent_dim": 64,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "corruption": "gaussian",      # gaussian | mask | salt_pepper
    "corruption_strength": 0.2,    # sigma / patch drop rate / pixel hit rate. 0 = vanilla AE

    # --- representation monitoring
    "knn_every_n_epochs": 5,       # the model-selection metric is logged this often
}


def load_config(yaml_path=None):
    """Return DEFAULTS updated with the values in the YAML file (if given)."""
    config = dict(DEFAULTS)
    if yaml_path is not None:
        with open(yaml_path) as file:
            overrides = yaml.safe_load(file) or {}
        unknown_keys = set(overrides) - set(DEFAULTS)
        if unknown_keys:
            # A typo like "latnt_dim" would otherwise silently run with the default.
            raise ValueError(f"Unknown config keys in {yaml_path}: {sorted(unknown_keys)}")
        config.update(overrides)
    return config
