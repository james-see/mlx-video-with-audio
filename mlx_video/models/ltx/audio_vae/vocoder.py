"""Vocoder for converting mel spectrograms to audio waveforms."""

import math
from typing import List

import mlx.core as mx
import mlx.nn as nn

from .resnet import LRELU_SLOPE, ResBlock1, ResBlock2, leaky_relu


class Vocoder(nn.Module):
    """
    Vocoder model for synthesizing audio from Mel spectrograms.
    Based on HiFi-GAN architecture.

    Args:
        resblock_kernel_sizes: List of kernel sizes for the residual blocks
        upsample_rates: List of upsampling rates
        upsample_kernel_sizes: List of kernel sizes for the upsampling layers
        resblock_dilation_sizes: List of dilation sizes for the residual blocks
        upsample_initial_channel: Initial number of channels for upsampling
        stereo: Whether to use stereo output
        resblock: Type of residual block to use ("1" or "2")
        output_sample_rate: Waveform sample rate
    """

    def __init__(
        self,
        resblock_kernel_sizes: List[int] | None = None,
        upsample_rates: List[int] | None = None,
        upsample_kernel_sizes: List[int] | None = None,
        resblock_dilation_sizes: List[List[int]] | None = None,
        upsample_initial_channel: int = 1024,
        stereo: bool = True,
        resblock: str = "1",
        output_sample_rate: int = 24000,
    ):
        super().__init__()

        # Initialize default values if not provided
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if upsample_rates is None:
            upsample_rates = [6, 5, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 15, 8, 4, 4]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.output_sample_rate = output_sample_rate
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.upsample_initial_channel = upsample_initial_channel

        in_channels = 128 if stereo else 64
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, kernel_size=7, stride=1, padding=3)

        resblock_class = ResBlock1 if resblock == "1" else ResBlock2

        # Upsampling layers using ConvTranspose1d
        self.ups = {}
        for i, (stride, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2**i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups[i] = nn.ConvTranspose1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            )

        # Residual blocks
        self.resblocks = {}
        block_idx = 0
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks[block_idx] = resblock_class(ch, kernel_size, tuple(dilations))
                block_idx += 1

        out_channels = 2 if stereo else 1
        final_channels = upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = nn.Conv1d(final_channels, out_channels, kernel_size=7, stride=1, padding=3)

        self.upsample_factor = math.prod(upsample_rates)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of the vocoder.
        Args:
            x: Input Mel spectrogram tensor. Can be either:
               - 3D: (batch_size, time, mel_bins) for mono - MLX format (N, L, C)
               - 4D: (batch_size, 2, time, mel_bins) for stereo - PyTorch format (N, C, H, W)
        Returns:
            Audio waveform tensor of shape (batch_size, out_channels, audio_length)
        """
        # Input: (batch, channels, time, mel_bins) from audio decoder
        # Transpose to (batch, channels, mel_bins, time)
        x = mx.transpose(x, (0, 1, 3, 2))

        if x.ndim == 4:  # stereo
            # x shape: (batch, 2, mel_bins, time)
            # Rearrange to (batch, 2*mel_bins, time)
            b, s, c, t = x.shape
            x = x.reshape(b, s * c, t)

        # MLX Conv1d expects (N, L, C), so transpose
        # Current: (batch, channels, time) -> (batch, time, channels)
        x = mx.transpose(x, (0, 2, 1))

        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            start = i * self.num_kernels
            end = start + self.num_kernels

            # Apply residual blocks and average their outputs
            block_outputs = []
            for idx in range(start, end):
                block_outputs.append(self.resblocks[idx](x))

            # Stack and mean
            x = mx.stack(block_outputs, axis=0)
            x = mx.mean(x, axis=0)

        # IMPORTANT: Use default leaky_relu slope (0.01), NOT LRELU_SLOPE (0.1)
        # PyTorch uses F.leaky_relu(x) which defaults to 0.01
        x = nn.leaky_relu(x)  # Default negative_slope=0.01
        x = self.conv_post(x)
        x = mx.tanh(x)

        # Transpose back to (batch, channels, time)
        x = mx.transpose(x, (0, 2, 1))

        return x
