"""Run configuration: defaults here, overridden by one YAML file per experiment.

Every run is `python train.py --config configs/<name>.yaml`. The YAML only lists
what differs from DEFAULTS, so each file reads as "what this experiment changes".
"""
import yaml

DEFAULTS = {
    # --- run identity and paths
    "run_name": "dae_default",
    "mode": "pretrain_dae",        # pretrain_dae | finetune | evaluate | robustness (see utils.build_model / train.py)
    "seed": 42,                    # fixed seed: same init and data order on every rerun
    "data_dir": "dataset/",        # cached Tiny-ImageNet tensors live here
    "output_dir": "outputs",       # checkpoints, TensorBoard logs, saved encoders (put on Drive in Colab)
    "results_dir": "results",      # small metric JSONs, committed to git

    # --- data
    "dataset": "tiny_imagenet",    # tiny_imagenet | cifar100 (transfer: evaluate / finetune on 64x64 CIFAR-100)
    "batch_size": 256,
    "train_fraction": 1.0,         # balanced fraction of the 90k train images (0.25 = proxy run)
    "num_workers": 0,              # data is in memory as tensors, workers would only add overhead

    # --- training
    "max_epochs": 100,             # a cap. With early stopping on, the rule below decides when training ends
    # Stop rule (None = off, train for max_epochs). Watches the validation hold-out, never the test split:
    # val_top1 for fine-tuning (higher is better), val_recon_loss for the autoencoder (lower is better).
    # Training stops after this many epochs without improvement, and the best epoch's weights are kept.
    "early_stopping_patience": None,
    "early_stopping_min_delta": 0.0,  # an improvement smaller than this does not reset the patience counter
    "precision": "auto",           # "16-mixed" on GPU, "32-true" on CPU
    "resume": True,                # continue from outputs/checkpoints/<run_name>/last.ckpt if present
    "fast_dev_run": False,         # one batch through every step: a wiring test, not a real run

    # --- model: denoising autoencoder
    "latent_dim": 64,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "corruption": "gaussian",      # gaussian | mask | salt_pepper
    "corruption_strength": 0.2,    # sigma / patch drop rate / pixel hit rate. 0 = vanilla AE

    # --- fine-tuning (mode "finetune"). encoder_path None = train from scratch (supervised baseline).
    # train_fraction doubles as the label fraction here: 0.01 / 0.1 / 1.0 of the balanced train set.
    "encoder_learning_rate": 1e-4, # gentle steps for pretrained weights. Set equal to learning_rate for scratch

    # --- evaluation (mode "evaluate": measure a trained or random encoder, no training)
    "encoder_path": None,          # outputs/encoders/<run>.pt to evaluate. None = random-init encoder baseline
    "eval_split": "val",           # "val" while tuning. "test" only for the final numbers
    "feature_layer": "conv",       # which layer is h: conv = pooled conv map (our choice, see report) | bottleneck = the Linear output
    "probe_seeds": [42],           # one linear probe per seed, reported as mean +- std
    "probe_epochs": 50,

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
