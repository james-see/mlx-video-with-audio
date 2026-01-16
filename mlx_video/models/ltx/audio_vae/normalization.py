"""Normalization layers for audio VAE."""

from enum import Enum

import mlx.core as mx
import mlx.nn as nn


class NormType(Enum):
    """Normalization layer types: GROUP (GroupNorm) or PIXEL (per-location RMS norm)."""

    GROUP = "group"
    PIXEL = "pixel"


class PixelNorm(nn.Module):
    """
    Per-pixel (per-location) RMS normalization layer.
    For each element along the chosen dimension, this layer normalizes the tensor
    by the root-mean-square of its values across that dimension:
        y = x / sqrt(mean(x^2, dim=dim, keepdim=True) + eps)
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        """
        Args:
            dim: Dimension along which to compute the RMS (typically channels).
            eps: Small constant added for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization along the configured dimension."""
        mean_sq = mx.mean(x**2, axis=self.dim, keepdims=True)
        rms = mx.sqrt(mean_sq + self.eps)
        return x / rms


def build_normalization_layer(
    in_channels: int, *, num_groups: int = 32, normtype: NormType = NormType.GROUP
) -> nn.Module:
    """
    Create a normalization layer based on the normalization type.
    Args:
        in_channels: Number of input channels
        num_groups: Number of groups for group normalization
        normtype: Type of normalization: "group" or "pixel"
    Returns:
        A normalization layer
    """
    if normtype == NormType.GROUP:
        return nn.GroupNorm(num_groups=num_groups, dims=in_channels, eps=1e-6, affine=True)
    if normtype == NormType.PIXEL:
        # For MLX channels-last format (B, H, W, C), normalize along channels (dim=-1)
        # PyTorch uses dim=1 for channels-first format (B, C, H, W)
        return PixelNorm(dim=-1, eps=1e-6)
    raise ValueError(f"Invalid normalization type: {normtype}")
