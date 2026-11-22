"""Representation evaluation: feature extraction and weighted cosine k-NN.

Also the linear probe, and run_evaluation (mode: evaluate), which measures one frozen
encoder with both. The collapse diagnostics (alignment, uniformity, std, effective rank)
are added in a later piece.
"""
import json
import os
import statistics

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from datamodule import NUM_CLASSES, build_datamodule
from model import build_encoder, build_feature_extractor


@torch.no_grad()
def extract_features(encoder, dataloader, device):
    """Run the frozen encoder over a dataloader, return (features, labels) on CPU.

    Works with or without the Trainer: the datamodule hook converts uint8 to float
    only inside a Trainer run, so the conversion is repeated here when needed.
    """
    was_training = encoder.training
    encoder.eval()  # no dropout / batch-norm updates while measuring a representation
    encoder.to(device)

    all_features, all_labels = [], []
    for images, labels in dataloader:
        images = images.to(device)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        all_features.append(encoder(images).cpu())
        all_labels.append(labels)

    encoder.train(was_training)
    return torch.cat(all_features), torch.cat(all_labels)


@torch.no_grad()
def knn_accuracy(bank_features, bank_labels, query_features, query_labels,
                 k=20, temperature=0.07, num_classes=200, chunk_size=1000):
    """Weighted cosine k-NN. Returns {"top1": ..., "top5": ...} in [0, 1].

    Each of the k nearest bank images votes for its class with weight
    exp(cosine / temperature), so close neighbors count much more than far ones.
    Raw cosine weights would let negative similarities cancel votes. Top-5 uses the
    five classes with the highest vote totals. No training is involved.
    """
    # After l2-normalisation, a dot product is the cosine similarity.
    bank = F.normalize(bank_features.float(), dim=1)
    queries = F.normalize(query_features.float(), dim=1)

    correct_top1, correct_top5 = 0, 0
    for start in range(0, len(queries), chunk_size):
        chunk = queries[start:start + chunk_size]
        similarity = chunk @ bank.T                                  # [chunk, bank_size]
        top_sim, top_idx = similarity.topk(k, dim=1)                 # k nearest bank images
        neighbor_labels = bank_labels[top_idx]                       # [chunk, k]
        vote_weight = torch.exp(top_sim / temperature)

        class_scores = torch.zeros(len(chunk), num_classes)
        class_scores.scatter_add_(1, neighbor_labels, vote_weight)   # sum weights per class

        true_labels = query_labels[start:start + chunk_size]
        top5_classes = class_scores.topk(min(5, num_classes), dim=1).indices
        correct_top1 += (top5_classes[:, 0] == true_labels).sum().item()
        correct_top5 += (top5_classes == true_labels[:, None]).any(dim=1).sum().item()

    return {"top1": correct_top1 / len(queries), "top5": correct_top5 / len(queries)}


def knn_by_layer(encoder, datamodule, device, layers=("bottleneck", "conv")):
    """Validation k-NN of an encoder on each candidate feature layer: {layer: {"top1", "top5"}}.

    The bank is the train images the datamodule serves, the queries are the 10k validation
    hold-out. Used during pre-training (KNNMonitor) and once at the end of a run, so the search
    itself shows which layer makes the better representation h.
    """
    results = {}
    for layer in layers:
        extractor = build_feature_extractor(encoder, layer)
        bank_features, bank_labels = extract_features(extractor, datamodule.train_eval_dataloader(), device)
        val_features, val_labels = extract_features(extractor, datamodule.val_dataloader(), device)
        results[layer] = knn_accuracy(bank_features, bank_labels, val_features, val_labels)
    return results


class LinearProbe(pl.LightningModule):
    """One linear layer on frozen, pre-extracted features.

    The encoder never changes, so its features are computed once and cached.
    The probe then trains in seconds instead of re-running the encoder every epoch.
    Features are standardised with train statistics: encoders output very different
    scales, and a probe on raw features would mostly measure that scale.
    """

    def __init__(self, feature_mean, feature_std, num_classes=200, learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["feature_mean", "feature_std"])
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)
        self.classifier = nn.Linear(len(feature_mean), num_classes)

    def forward(self, features):
        return self.classifier((features - self.feature_mean) / self.feature_std)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        loss = F.cross_entropy(self(features), labels)
        self.log("probe_train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)


@torch.no_grad()
def top_k_accuracy(model, features, labels):
    """Return {"top1", "top5"} of a classifier on cached features."""
    model.eval()
    logits = model(features.to(model.device))
    top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices.cpu()
    return {"top1": (top5[:, 0] == labels).float().mean().item(),
            "top5": (top5 == labels[:, None]).any(dim=1).float().mean().item()}


def train_linear_probe(train_features, train_labels, num_classes=200, epochs=50, batch_size=1024,
                       learning_rate=1e-3, weight_decay=1e-4, seed=42):
    """Train a linear probe on cached train features and return the fitted LinearProbe.

    The seed controls the probe's init and batch order, so repeating with several
    seeds gives the mean +- std the report needs.
    """
    pl.seed_everything(seed, workers=True)
    train_features = train_features.float()
    probe = LinearProbe(train_features.mean(0), train_features.std(0) + 1e-6,
                        num_classes, learning_rate, weight_decay)
    loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
    trainer = pl.Trainer(max_epochs=epochs, accelerator="auto", devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False)
    trainer.fit(probe, loader)
    return probe


def linear_probe_accuracy(train_features, train_labels, test_features, test_labels, **probe_settings):
    """Train a linear probe on train features, return its top-1/top-5 on test features."""
    probe = train_linear_probe(train_features, train_labels, **probe_settings)
    return top_k_accuracy(probe, test_features.float(), test_labels)


def run_evaluation(config):
    """Evaluate an encoder on the validation or test split, save results/<run_name>.json."""
    pl.seed_everything(config["seed"], workers=True)  # also fixes the random-init weights
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datamodule = build_datamodule(config)
    datamodule.prepare_data()
    datamodule.setup()

    encoder = build_encoder(config["latent_dim"])
    if config["encoder_path"] is not None:
        encoder.load_state_dict(torch.load(config["encoder_path"], map_location="cpu"))
    extractor = build_feature_extractor(encoder, config["feature_layer"])

    eval_loader = datamodule.test_dataloader() if config["eval_split"] == "test" else datamodule.val_dataloader()
    train_features, train_labels = extract_features(extractor, datamodule.train_eval_dataloader(), device)
    eval_features, eval_labels = extract_features(extractor, eval_loader, device)

    num_classes = NUM_CLASSES[config["dataset"]]
    knn = knn_accuracy(train_features, train_labels, eval_features, eval_labels, num_classes=num_classes)
    probes = [linear_probe_accuracy(train_features, train_labels, eval_features, eval_labels,
                                    num_classes=num_classes, epochs=config["probe_epochs"], seed=seed)
              for seed in config["probe_seeds"]]
    probe_top1 = [probe["top1"] for probe in probes]
    metrics = {
        "knn_top1": knn["top1"], "knn_top5": knn["top5"],
        "probe_top1_mean": statistics.mean(probe_top1),
        "probe_top1_std": statistics.pstdev(probe_top1),  # 0 with a single seed
        "probe_top5_mean": statistics.mean(probe["top5"] for probe in probes),
        "feature_dim": train_features.shape[1],
    }

    os.makedirs(config["results_dir"], exist_ok=True)
    with open(os.path.join(config["results_dir"], f"{config['run_name']}.json"), "w") as file:
        json.dump({"config": config, "metrics": metrics}, file, indent=2)
    return metrics
