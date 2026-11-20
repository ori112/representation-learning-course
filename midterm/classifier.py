"""Supervised classifier on top of an encoder: fine-tuning and the from-scratch baseline.

The same class covers both. With pretrained encoder weights it is fine-tuning, with
a random-init encoder it is the supervised baseline. The architecture and the training
recipe are identical, so any accuracy gap comes from the pretraining alone.
"""
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim

from evaluate import build_feature_extractor


def top_k_correct(logits, labels, k):
    """Number of samples whose true class is among the k highest logits."""
    top_classes = logits.topk(min(k, logits.shape[1]), dim=1).indices
    return (top_classes == labels[:, None]).any(dim=1).float().sum()


class EncoderClassifier(pl.LightningModule):
    """encoder (trainable) -> linear head over the classes.

    The encoder gets a smaller learning rate than the head: pretrained weights should
    move gently, while the new head starts from scratch and needs larger steps. For the
    from-scratch baseline both rates are set equal in the config.
    """

    def __init__(self, encoder, feature_layer="bottleneck", num_classes=200,
                 learning_rate=1e-3, encoder_learning_rate=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.features = build_feature_extractor(encoder, feature_layer)
        with torch.no_grad():  # find the feature size by pushing one dummy image through
            feature_dim = self.features(torch.zeros(1, 3, 64, 64)).shape[1]
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, images):
        return self.head(self.features(images))

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss = F.cross_entropy(self(images), labels)
        self.log("train_loss", loss)
        return loss

    def _evaluate_batch(self, batch, prefix):
        """Log top-1/top-5 accuracy of a batch, averaged over the whole epoch."""
        images, labels = batch
        logits = self(images)
        for k in (1, 5):
            self.log(f"{prefix}_top{k}", top_k_correct(logits, labels, k) / len(labels),
                     on_epoch=True, on_step=False, prog_bar=(k == 1), batch_size=len(labels))

    def validation_step(self, batch, batch_idx):
        self._evaluate_batch(batch, "val")

    def test_step(self, batch, batch_idx):
        self._evaluate_batch(batch, "test")

    def configure_optimizers(self):
        return optim.Adam([
            {"params": self.features.parameters(), "lr": self.hparams.encoder_learning_rate},
            {"params": self.head.parameters(), "lr": self.hparams.learning_rate},
        ], weight_decay=self.hparams.weight_decay)
