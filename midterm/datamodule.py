"""Tiny-ImageNet and CIFAR-100 data: one-time cache to uint8 tensors, seeded splits,
and the LightningDataModules that serve them."""
import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

SPLIT_SEED = 42  # same seed everywhere, so every model sees the same splits
NUM_CLASSES = {"tiny_imagenet": 200, "cifar100": 100}


def load_tiny_imagenet(cache_dir):
    """Return uint8 image tensors and labels for the train and valid splits.

    Decoding 110k PIL images on Colab's 2 CPU cores is slow, so it is done once
    and saved as a single .pt file (about 1.3 GB). Later runs just torch.load it.
    """
    cache_path = os.path.join(cache_dir, "tiny_imagenet_uint8.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    from datasets import load_dataset  # imported here so cached runs skip the dependency cost

    hf_dataset = load_dataset("zh-plus/tiny-imagenet")
    cached = {}
    for split_name in ("train", "valid"):
        split = hf_dataset[split_name]
        # A few images are grayscale, so force RGB to get a uniform 3x64x64 shape.
        images = np.stack([np.array(example["image"].convert("RGB")) for example in split])
        cached[split_name] = {
            "images": torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(),  # [N,3,64,64] uint8
            "labels": torch.tensor(split["label"], dtype=torch.long),
        }
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(cached, cache_path)
    return cached


def load_cifar100_at_64(cache_dir):
    """Return CIFAR-100 as 64x64 uint8 tensors, in the same layout as load_tiny_imagenet.

    CIFAR-100 is the transfer dataset. Its images are 32x32, so they are upsampled to 64x64
    (bilinear) to fit the encoder, which was built for Tiny-ImageNet. The official test split
    is stored under "valid" so both datasets expose the same keys.
    """
    cache_path = os.path.join(cache_dir, "cifar100_64_uint8.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    import torch.nn.functional as functional
    from torchvision.datasets import CIFAR100  # imported here so Tiny-ImageNet runs skip it

    cached = {}
    for split_name, is_train in (("train", True), ("valid", False)):
        dataset = CIFAR100(root=cache_dir, train=is_train, download=True)
        images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous()  # [N,3,32,32] uint8
        resized = torch.cat([
            functional.interpolate(chunk.float(), size=64, mode="bilinear", align_corners=False)
            .round().clamp(0, 255).to(torch.uint8)
            for chunk in images.split(5000)  # chunked: a float copy of all 50k images is 2.4 GB
        ])
        cached[split_name] = {"images": resized, "labels": torch.tensor(dataset.targets, dtype=torch.long)}
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(cached, cache_path)
    return cached


def split_train_holdout(labels, holdout_per_class=50, seed=SPLIT_SEED):
    """Split train indices into (train, validation), holding out N images per class.

    A class-balanced hold-out keeps validation k-NN accuracy comparable across
    classes. The test set is the official valid split and is never used here.
    """
    generator = torch.Generator().manual_seed(seed)
    train_indices, holdout_indices = [], []
    for class_id in labels.unique():
        class_indices = (labels == class_id).nonzero().squeeze(1)
        shuffled = class_indices[torch.randperm(len(class_indices), generator=generator)]
        holdout_indices.append(shuffled[:holdout_per_class])
        train_indices.append(shuffled[holdout_per_class:])
    return torch.cat(train_indices).sort().values, torch.cat(holdout_indices).sort().values


def balanced_label_subset(labels, indices, fraction, seed=SPLIT_SEED):
    """Pick the same fraction of images from every class, out of the given indices.

    The seed is fixed, so 1% / 10% subsets are identical for every model, as the
    assignment requires. Subsets are nested: the 1% images are inside the 10% set.
    """
    generator = torch.Generator().manual_seed(seed)
    chosen = []
    for class_id in labels[indices].unique():
        class_indices = indices[labels[indices] == class_id]
        shuffled = class_indices[torch.randperm(len(class_indices), generator=generator)]
        count = max(1, round(fraction * len(class_indices)))
        chosen.append(shuffled[:count])
    return torch.cat(chosen).sort().values


class TinyImageNetDataModule(pl.LightningDataModule):
    """Train / validation / test loaders for Tiny-ImageNet at 64x64.

    - train: the 90k images left after the hold-out (or a balanced fraction of them)
    - val:   the 10k hold-out, used for all tuning
    - test:  the official valid split, used only for final numbers

    Labels are always returned. Self-supervised pre-training simply ignores them,
    and the evaluation modes (k-NN, linear probe, fine-tuning) use them.
    """

    def __init__(self, data_dir="dataset/", batch_size=256, train_fraction=1.0, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # Balanced fraction of the train split: 0.25 for proxy pre-training runs,
        # 0.01 / 0.1 for the label-efficiency fine-tuning runs.
        self.train_fraction = train_fraction
        # The data already sits in memory as tensors, so worker processes add overhead
        # and no speed-up. Keep 0 unless a profile says otherwise.
        self.num_workers = num_workers

    holdout_per_class = 50

    def load_cache(self):
        """Return {"train": {...}, "valid": {...}} uint8 tensors. Subclasses swap the dataset."""
        return load_tiny_imagenet(self.data_dir)

    def prepare_data(self):
        # Runs once on one process: download and build the cache.
        self.load_cache()

    def setup(self, stage=None):
        cached = self.load_cache()
        train_images, train_labels = cached["train"]["images"], cached["train"]["labels"]

        train_pool, holdout = split_train_holdout(train_labels, self.holdout_per_class)
        # The fraction is taken from the train pool only, so val images never leak in.
        train_indices = balanced_label_subset(train_labels, train_pool, self.train_fraction)

        self.train_ds = TensorDataset(train_images[train_indices], train_labels[train_indices])
        self.val_ds = TensorDataset(train_images[holdout], train_labels[holdout])
        self.test_ds = TensorDataset(cached["valid"]["images"], cached["valid"]["labels"])

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Convert uint8 [0,255] to float [0,1] on the GPU, after the batch is moved.

        Moving uint8 is 4x cheaper than moving float32, and the conversion costs
        nothing on the GPU.
        """
        images, labels = batch
        return images.float() / 255.0, labels

    def _loader(self, dataset, shuffle, drop_last=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self):
        # drop_last: the contrastive loss treats the rest of the batch as negatives,
        # so a short last batch would change the number of negatives.
        return self._loader(self.train_ds, shuffle=True, drop_last=True)

    def train_eval_dataloader(self):
        """The same train images, unshuffled and complete. Used as the k-NN bank."""
        return self._loader(self.train_ds, shuffle=False)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False)


class Cifar100DataModule(TinyImageNetDataModule):
    """CIFAR-100 resized to 64x64: the transfer dataset.

    Same protocol as Tiny-ImageNet: 50 images per class (5k) held out of the 50k train
    images for validation, the official 10k test split for final numbers only, and the
    same balanced label fractions. The encoder is pre-trained on Tiny-ImageNet, never on this.
    """

    def load_cache(self):
        return load_cifar100_at_64(self.data_dir)


def build_datamodule(config):
    """Create the DataModule named by config["dataset"] (tiny_imagenet | cifar100)."""
    datamodules = {"tiny_imagenet": TinyImageNetDataModule, "cifar100": Cifar100DataModule}
    if config["dataset"] not in datamodules:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    return datamodules[config["dataset"]](
        data_dir=config["data_dir"], batch_size=config["batch_size"],
        train_fraction=config["train_fraction"], num_workers=config["num_workers"])
