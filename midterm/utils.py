from model import DenoisingAutoEncoder


def build_model(config):
    """Create the LightningModule for config["mode"]. Modes are added as they are built."""
    if config["mode"] == "pretrain_dae":
        return DenoisingAutoEncoder(
            latent_dim=config["latent_dim"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            corruption=config["corruption"],
            corruption_strength=config["corruption_strength"],
        )
    raise ValueError(f"Unknown mode: {config['mode']}")
