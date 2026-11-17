"""Input corruptions for the denoising autoencoder (and later, the robustness test).

All functions take a float image batch in [0, 1] shaped [B, 3, H, W] and return a
corrupted copy of the same shape and range. Strength 0 returns the input unchanged,
which is how the plain (vanilla) autoencoder baseline is obtained.
"""
import torch
import torch.nn.functional as F

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
