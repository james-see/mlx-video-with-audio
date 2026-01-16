"""Audio processing utilities for audio VAE."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class AudioLatentShape:
    """Shape descriptor for audio latent representations."""

    batch: int
    channels: int
    frames: int
    mel_bins: int


class PerChannelStatistics(nn.Module):
    """
    Per-channel statistics for normalizing and denormalizing the latent representation.
    This statistics is computed over the entire dataset and stored in model's checkpoint.
    """

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        # Initialize buffers - will be loaded from weights
        # Using underscores for MLX compatibility with weight loading
        self._std_of_means = mx.ones((latent_channels,))
        self._mean_of_means = mx.zeros((latent_channels,))

    def un_normalize(self, x: mx.array) -> mx.array:
        """Denormalize latent representation."""
        # Broadcast statistics to match x shape
        # x shape: (B, C, ...) or (B, ..., C)
        std = self._std_of_means.astype(x.dtype)
        mean = self._mean_of_means.astype(x.dtype)
        return (x * std) + mean

    def normalize(self, x: mx.array) -> mx.array:
        """Normalize latent representation."""
        std = self._std_of_means.astype(x.dtype)
        mean = self._mean_of_means.astype(x.dtype)
        return (x - mean) / std


class AudioPatchifier:
    """
    Audio patchifier for converting between audio latents and patches.
    Combines channels and mel_bins dimensions for per-channel statistics.
    """

    def __init__(
        self,
        patch_size: int = 1,
        audio_latent_downsample_factor: int = 4,
        sample_rate: int = 16000,
        hop_length: int = 160,
        is_causal: bool = True,
    ):
        self.patch_size = patch_size
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.is_causal = is_causal

    def patchify(self, x: mx.array) -> mx.array:
        """Convert audio latents to patches.

        Input shape: (B, T, F, C) in MLX format (channels last)
        Output shape: (B, T, C*F) - flattened for per-channel statistics

        The output order is (c f) to match PyTorch's "b c t f -> b t (c f)".
        """
        # x shape: (B, T, F, C) e.g., (1, 68, 16, 8)
        b, t, f, c = x.shape
        # Transpose to (B, T, C, F) for correct (c f) ordering
        x = mx.transpose(x, (0, 1, 3, 2))
        # Reshape to (B, T, C*F) e.g., (1, 68, 128)
        return x.reshape(b, t, c * f)

    def unpatchify(self, x: mx.array, latent_shape: AudioLatentShape) -> mx.array:
        """Convert patches back to audio latents.

        Input shape: (B, T, C*F)
        Output shape: (B, T, F, C) in MLX format

        Reverses patchify's "b t (c f) -> b c t f" then transposes to MLX format.
        """
        # x shape: (B, T, C*F) e.g., (1, 68, 128)
        b, t, cf = x.shape
        c = latent_shape.channels
        f = latent_shape.mel_bins
        # Reshape to (B, T, C, F)
        x = x.reshape(b, t, c, f)
        # Transpose to MLX format (B, T, F, C)
        return mx.transpose(x, (0, 1, 3, 2))
