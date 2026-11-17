"""Models for the midterm. Built piece by piece.

The small encoder is shared by the autoencoder, SimCLR and the baselines.
"""
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim

from corruptions import corrupt


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


def build_decoder(latent_dim=64):
    """Light decoder g_phi: the mirror image of the encoder, discarded after training.

    A linear layer expands h back to 64x16x16, and two transposed convolutions undo
    the two stride-2 convolutions (16 -> 32 -> 64). output_padding=1 makes the sizes
    land exactly on 32 and 64. Sigmoid keeps pixels in (0, 1), the same range as the
    [0, 1] images the datamodule produces, so the reconstruction target is reachable.
    It is kept small on purpose: a strong decoder would do the work and leave the
    encoder, which is the thing we evaluate, with less to learn.
    """
    return nn.Sequential(
        nn.Linear(latent_dim, 64 * 16 * 16),                                                # [B, latent] -> [B, 16384]
        nn.ReLU(),
        nn.Unflatten(1, (64, 16, 16)),                                                      # -> [B, 64, 16, 16]
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> [B, 32, 32, 32]
        nn.ReLU(),
        nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # -> [B, 3, 64, 64]
        nn.Sigmoid(),
    )


class DenoisingAutoEncoder(pl.LightningModule):
    """Denoising autoencoder: reconstruct the clean image x from a corrupted x_tilde.

    loss = MSE(x, g(f(x_tilde))). Corruption is applied in training only. Validation
    reconstructs clean images from clean images, so the reconstruction error is
    comparable across trials with different corruptions (needed for the A2 plot).
    corruption_strength = 0 turns this into the plain autoencoder baseline.
    Labels in the batch are never used.
    """

    def __init__(self, latent_dim=64, learning_rate=1e-3, weight_decay=0.0,
                 corruption="gaussian", corruption_strength=0.2):
        super().__init__()
        self.save_hyperparameters()  # logged by TensorBoard, restored from checkpoints
        self.encoder = build_encoder(latent_dim)
        self.decoder = build_decoder(latent_dim)

    def forward(self, images):
        return self.decoder(self.encoder(images))

    def training_step(self, batch, batch_idx):
        clean_images, _ = batch  # labels ignored: this is self-supervised
        corrupted = corrupt(clean_images, self.hparams.corruption, self.hparams.corruption_strength)
        loss = F.mse_loss(self(corrupted), clean_images)  # target is the CLEAN image
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        clean_images, _ = batch
        reconstruction = self(clean_images)
        self.log("val_recon_loss", F.mse_loss(reconstruction, clean_images), on_epoch=True, prog_bar=True)
        if batch_idx == 0 and self.logger is not None:
            self._log_reconstruction_grid(clean_images[:8])

    def _log_reconstruction_grid(self, clean_images):
        """TensorBoard image: row 1 clean, row 2 what the encoder saw in training, row 3 output."""
        corrupted = corrupt(clean_images, self.hparams.corruption, self.hparams.corruption_strength)
        with torch.no_grad():
            output = self(corrupted)
        grid = torchvision.utils.make_grid(torch.cat([clean_images, corrupted, output]), nrow=len(clean_images))
        self.logger.experiment.add_image("reconstruction (clean / corrupted / output)", grid, self.current_epoch)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
