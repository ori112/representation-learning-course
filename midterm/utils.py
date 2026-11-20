import torch

from classifier import EncoderClassifier
from model import DenoisingAutoEncoder, build_encoder


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
    if config["mode"] == "finetune":
        encoder = build_encoder(config["latent_dim"])
        if config["encoder_path"] is not None:  # None = random init: the supervised baseline
            encoder.load_state_dict(torch.load(config["encoder_path"], map_location="cpu"))
        return EncoderClassifier(
            encoder,
            feature_layer=config["feature_layer"],
            learning_rate=config["learning_rate"],
            encoder_learning_rate=config["encoder_learning_rate"],
            weight_decay=config["weight_decay"],
        )
    raise ValueError(f"Unknown mode: {config['mode']}")
