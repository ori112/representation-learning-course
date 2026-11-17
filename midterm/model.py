"""Models for the midterm. Built piece by piece.

Piece 1: the small encoder shared by the autoencoder, SimCLR and the baselines.
"""
from torch import nn


def build_encoder(latent_dim=64):
    """The course small encoder, adapted from 1x28x28 (MNIST) to 3x64x64 (Tiny-ImageNet).

    Two stride-2 convolutions halve the image twice (64 -> 32 -> 16), and a 3x3 kernel
    with padding 1 does that exactly. Channels grow 3 -> 32 -> 64 as the map shrinks.
    A linear layer then compresses 64*16*16 numbers into a short vector h.

    It is a plain nn.Sequential on purpose: encoder[:4] returns the conv feature map
    [B, 64, 16, 16], so the choice of which layer to use as h can be tested by slicing.
    There is no batch-norm, matching the course encoder.
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # [B, 3, 64, 64]  -> [B, 32, 32, 32]
        nn.ReLU(),                                              # same shape
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 32, 32, 32] -> [B, 64, 16, 16]
        nn.ReLU(),                                              # same shape
        nn.Flatten(),                                           # [B, 64, 16, 16] -> [B, 16384]
        nn.Linear(64 * 16 * 16, latent_dim),                    # [B, 16384]      -> [B, latent_dim]
    )
