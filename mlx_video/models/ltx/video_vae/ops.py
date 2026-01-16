"""Operations for Video VAE."""

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn


def patchify(x: mx.array, patch_size_hw: int = 4, patch_size_t: int = 1) -> mx.array:
    """Convert video to patches.

    Moves spatial pixels from H, W dimensions to channel dimension.

    Args:
        x: Input tensor of shape (B, C, F, H, W)
        patch_size_hw: Spatial patch size
        patch_size_t: Temporal patch size

    Returns:
        Patched tensor of shape (B, C * patch_size_hw^2, F, H/patch_size_hw, W/patch_size_hw)
    """
    b, c, f, h, w = x.shape

    # Check dimensions are divisible
    assert h % patch_size_hw == 0 and w % patch_size_hw == 0
    assert f % patch_size_t == 0

    # New dimensions
    new_h = h // patch_size_hw
    new_w = w // patch_size_hw
    new_f = f // patch_size_t
    new_c = c * patch_size_hw * patch_size_hw * patch_size_t

    # Reshape: (B, C, F, H, W) -> (B, C, F/pt, pt, H/ph, ph, W/pw, pw)
    x = mx.reshape(x, (b, c, new_f, patch_size_t, new_h, patch_size_hw, new_w, patch_size_hw))

    # Permute: (B, C, F', pt, H', ph, W', pw) -> (B, C, pt, pw, ph, F', H', W')
    # PyTorch einops uses (c, p, r, q) = (c, temporal, width, height), so we need pw before ph
    x = mx.transpose(x, (0, 1, 3, 7, 5, 2, 4, 6))

    # Reshape: (B, C, pt, pw, ph, F', H', W') -> (B, C*pt*pw*ph, F', H', W')
    x = mx.reshape(x, (b, new_c, new_f, new_h, new_w))

    return x


def unpatchify(x: mx.array, patch_size_hw: int = 4, patch_size_t: int = 1) -> mx.array:
    """Convert patches back to video.

    Inverse of patchify - moves pixels from channel dimension back to spatial.
    Matches PyTorch einops: "b (c p r q) f h w -> b c (f p) (h q) (w r)"
    where p=patch_size_t, r=patch_size_hw (width), q=patch_size_hw (height)

    Args:
        x: Patched tensor of shape (B, C * patch_size_hw^2, F, H, W)
        patch_size_hw: Spatial patch size
        patch_size_t: Temporal patch size

    Returns:
        Video tensor of shape (B, C, F * patch_size_t, H * patch_size_hw, W * patch_size_hw)
    """
    b, c_packed, f, h, w = x.shape

    # Calculate original channel count
    c = c_packed // (patch_size_hw * patch_size_hw * patch_size_t)

    # Reshape: (B, C*pt*pr*pq, F, H, W) -> (B, C, pt, pr, pq, F, H, W)
    # where pt=temporal, pr=width_patch (r), pq=height_patch (q)
    # Channel layout from PyTorch is (c, p, r, q) = (c, temporal, width, height)
    x = mx.reshape(x, (b, c, patch_size_t, patch_size_hw, patch_size_hw, f, h, w))

    # Permute to interleave patches with spatial dims:
    # (B, C, pt, pr, pq, F, H, W) -> (B, C, F, pt, H, pq, W, pr)

    x = mx.transpose(x, (0, 1, 5, 2, 6, 4, 7, 3))

    # Reshape: (B, C, F, pt, H, pq, W, pr) -> (B, C, F*pt, H*pq, W*pr)
    x = mx.reshape(x, (b, c, f * patch_size_t, h * patch_size_hw, w * patch_size_hw))

    return x


class PerChannelStatistics(nn.Module):

    def __init__(self, latent_channels: int = 128):

        super().__init__()
        self.latent_channels = latent_channels

        # Learnable per-channel mean and std
        self.mean = mx.zeros((latent_channels,))
        self.std = mx.ones((latent_channels,))

    def normalize(self, x: mx.array) -> mx.array:
        """Normalize latents using per-channel statistics.

        Args:
            x: Input tensor of shape (B, C, ...)

        Returns:
            Normalized tensor
        """
        # Expand mean and std for broadcasting: (C,) -> (1, C, 1, 1, 1)
        mean = self.mean.reshape(1, -1, 1, 1, 1)
        std = self.std.reshape(1, -1, 1, 1, 1)

        return (x - mean) / std

    def un_normalize(self, x: mx.array) -> mx.array:
        """Denormalize latents using per-channel statistics.

        Args:
            x: Normalized tensor of shape (B, C, ...)

        Returns:
            Denormalized tensor
        """
        mean = self.mean.reshape(1, -1, 1, 1, 1)
        std = self.std.reshape(1, -1, 1, 1, 1)

        return x * std + mean
