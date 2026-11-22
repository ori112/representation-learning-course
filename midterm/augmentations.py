import torch
import torch.nn.functional as F
from torchvision.transforms import v2


class ContrastiveTransformations:
    """Apply one transform twice and return the two views.

    Each call samples new random parameters, so xi and xj differ.
    The base transform is whatever policy you pass in.
    """

    def __init__(self, base_transforms):
        self.base_transforms = base_transforms

    def __call__(self, x):
        xi = self.base_transforms(x)
        xj = self.base_transforms(x)
        return xi, xj


def build_simclr_transform():
    """One SimCLR view of a 28x28 grayscale image.

    Same steps as the notebook: crop, horizontal flip, color jitter, grayscale,
    and a gaussian blur. Blur kernels have to be odd, and 10% of 28 is 2, so
    the kernel is 3. Normalization matches the other MNIST transforms.
    """
    color_jitter = v2.ColorJitter(0.8, 0.8, 0.8, 0.2)
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(size=28),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([color_jitter], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            v2.Normalize((0.1307,), (0.3081,)),
        ]
    )


def build_mnist_transforms():
    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.Normalize((0.1307,), (0.3081,)),
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return train_transform, eval_transform


# ---------------------------------------------------------------------------
# Input corruptions for the denoising autoencoder and the robustness test.
# All functions take a float image batch in [0, 1] shaped [B, 3, H, W] and return a
# corrupted copy of the same shape and range. Strength 0 returns the input unchanged,
# which is how the plain (vanilla) autoencoder baseline is obtained.
# ---------------------------------------------------------------------------

PATCH_SIZE = 8  # masking removes 8x8 blocks: 64/8 = 8x8 grid of patches


def gaussian_noise(images, strength):
    """Add N(0, strength^2) noise, then clip back to the valid pixel range."""
    return (images + strength * torch.randn_like(images)).clamp(0.0, 1.0)


def patch_masking(images, strength):
    """Zero out whole 8x8 patches, each one dropped with probability `strength`.

    Patches rather than single pixels: a single missing pixel is easy to fill in
    from its neighbors, but a missing patch forces the encoder to use context.
    """
    batch, _, height, width = images.shape
    patch_grid = torch.rand(batch, 1, height // PATCH_SIZE, width // PATCH_SIZE, device=images.device) < strength
    pixel_mask = F.interpolate(patch_grid.float(), scale_factor=PATCH_SIZE, mode="nearest")
    return images * (1.0 - pixel_mask)


def salt_and_pepper(images, strength):
    """Set a fraction `strength` of pixel locations to black or white (50/50).

    The same location is hit in all three channels, so the damage looks like
    real salt-and-pepper speckle and not like random colour noise.
    """
    batch, _, height, width = images.shape
    hit = torch.rand(batch, 1, height, width, device=images.device) < strength
    salt = torch.rand(batch, 1, height, width, device=images.device) < 0.5  # white or black
    return torch.where(hit, salt.float().expand_as(images), images)


CORRUPTIONS = {
    "gaussian": gaussian_noise,
    "mask": patch_masking,
    "salt_pepper": salt_and_pepper,
}


def corrupt(images, kind, strength):
    """Apply the named corruption. Strength 0 is a no-op for every kind."""
    if strength == 0:
        return images
    return CORRUPTIONS[kind](images, strength)
