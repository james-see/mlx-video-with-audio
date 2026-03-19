"""Vocoder for converting mel spectrograms to audio waveforms."""

import math
from typing import List

import mlx.core as mx
import mlx.nn as nn

from .resnet import (
    AMPBlock1,
    LRELU_SLOPE,
    ResBlock1,
    ResBlock2,
    SnakeBeta,
    leaky_relu,
)


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
        self.conv_pre = nn.Conv1d(
            in_channels,
            upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        resblock_class = ResBlock1 if resblock == "1" else ResBlock2

        # Upsampling layers using ConvTranspose1d
        self.ups = {}
        for i, (stride, kernel_size) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes)
        ):
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
            for kernel_size, dilations in zip(
                resblock_kernel_sizes, resblock_dilation_sizes
            ):
                self.resblocks[block_idx] = resblock_class(
                    ch, kernel_size, tuple(dilations)
                )
                block_idx += 1

        out_channels = 2 if stereo else 1
        final_channels = upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = nn.Conv1d(
            final_channels, out_channels, kernel_size=7, stride=1, padding=3
        )

        self.upsample_factor = math.prod(upsample_rates)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of the vocoder.
        Args:
            x: Input Mel spectrogram tensor. Can be either:
               - 3D: (batch_size, time, mel_bins) for mono
                 MLX format (N, L, C)
               - 4D: (batch_size, 2, time, mel_bins) for stereo
                 PyTorch format (N, C, H, W)
        Returns:
            Audio waveform tensor of shape
            (batch_size, out_channels, audio_length)
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


class _STFTBasis(nn.Module):
    """Checkpoint-compatible holder for STFT basis tensors."""

    def __init__(self):
        super().__init__()
        self.forward_basis = mx.zeros((514, 1, 512), dtype=mx.float32)
        self.inverse_basis = mx.zeros((514, 1, 512), dtype=mx.float32)


class _MelSTFT(nn.Module):
    """Checkpoint-compatible holder for mel/STFT tensors."""

    def __init__(self):
        super().__init__()
        self.mel_basis = mx.zeros((64, 257), dtype=mx.float32)
        self.stft_fn = _STFTBasis()


class BigVGANVocoder(nn.Module):
    """BigVGAN-style vocoder used by distilled checkpoints."""

    def __init__(
        self,
        resblock_kernel_sizes: List[int] | None = None,
        upsample_rates: List[int] | None = None,
        upsample_kernel_sizes: List[int] | None = None,
        resblock_dilation_sizes: List[List[int]] | None = None,
        upsample_initial_channel: int = 1536,
        stereo: bool = True,
        output_sample_rate: int = 24000,
        use_tanh_at_final: bool = False,
        use_bias_at_final: bool = False,
        apply_final_activation: bool = True,
    ):
        super().__init__()

        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if upsample_rates is None:
            upsample_rates = [5, 2, 2, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [11, 4, 4, 4, 4, 4]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.output_sample_rate = output_sample_rate
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.use_tanh_at_final = use_tanh_at_final
        self.apply_final_activation = apply_final_activation

        in_channels = 128 if stereo else 64
        self.conv_pre = nn.Conv1d(
            in_channels,
            upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.ups = {}
        for i, (stride, kernel_size) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes)
        ):
            in_ch = upsample_initial_channel // (2**i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups[i] = nn.ConvTranspose1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            )

        self.resblocks = {}
        block_idx = 0
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(
                resblock_kernel_sizes, resblock_dilation_sizes
            ):
                self.resblocks[block_idx] = AMPBlock1(ch, kernel_size, tuple(dilations))
                block_idx += 1

        final_channels = upsample_initial_channel // (2**self.num_upsamples)
        self.act_post = SnakeBeta(final_channels)
        out_channels = 2 if stereo else 1
        self.conv_post = nn.Conv1d(
            final_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=use_bias_at_final,
        )

        # Optional checkpoint-only tensors from distilled vocoder checkpoints.
        self.mel_stft = _MelSTFT()

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 1, 3, 2))

        if x.ndim == 4:  # stereo
            b, s, c, t = x.shape
            x = x.reshape(b, s * c, t)

        x = mx.transpose(x, (0, 2, 1))
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            start = i * self.num_kernels
            end = start + self.num_kernels
            block_outputs = [self.resblocks[idx](x) for idx in range(start, end)]
            x = mx.mean(mx.stack(block_outputs, axis=0), axis=0)

        x = self.act_post(x)
        x = self.conv_post(x)
        if self.apply_final_activation:
            if self.use_tanh_at_final:
                x = mx.tanh(x)
            else:
                x = mx.clip(x, -1.0, 1.0)
        x = mx.transpose(x, (0, 2, 1))
        return x


class VocoderWithBWE(nn.Module):
    """Checkpoint-compatible vocoder + BWE wrapper.

    Notes:
    - Mirrors upstream module structure so `bwe_generator.*` and `mel_stft.*`
      tensors from checkpoints are loaded and used at runtime.
    - Uses an STFT-basis projection (from checkpoint tensors) to compute mel
      features for the BWE generator.
    """

    def __init__(
        self,
        vocoder: BigVGANVocoder,
        bwe_generator: BigVGANVocoder,
        input_sampling_rate: int = 16000,
        output_sampling_rate: int = 48000,
        hop_length: int = 80,
        win_length: int = 512,
    ):
        super().__init__()
        self.vocoder = vocoder
        self.bwe_generator = bwe_generator
        self.mel_stft = _MelSTFT()
        self.input_sampling_rate = input_sampling_rate
        self.output_sampling_rate = output_sampling_rate
        self.hop_length = hop_length
        self.win_length = win_length

    def _compute_mel(self, audio: mx.array) -> mx.array:
        """Compute log-mel from waveform using checkpoint STFT/mel bases.

        Args:
            audio: (B, C, T)
        Returns:
            (B, C, n_mels, T_frames)
        """
        b, c, t = audio.shape
        x = mx.reshape(audio, (b * c, t))

        # Upstream uses causal left-only pad: win_length - hop_length.
        left_pad = max(0, self.win_length - self.hop_length)
        x = mx.pad(x, [(0, 0), (left_pad, 0)])

        total = x.shape[1]
        if total < self.win_length:
            x = mx.pad(x, [(0, 0), (0, self.win_length - total)])
            total = x.shape[1]
        n_frames = max(1, (total - self.win_length) // self.hop_length + 1)

        basis = self.mel_stft.stft_fn.forward_basis[:, 0, :]  # (2*n_freq, win)
        n_freq = basis.shape[0] // 2
        mel_basis = self.mel_stft.mel_basis  # (n_mels, n_freq)

        frames = []
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.win_length
            seg = x[:, start:end]  # (B*C, win)
            if seg.shape[1] < self.win_length:
                seg = mx.pad(seg, [(0, 0), (0, self.win_length - seg.shape[1])])
            # (2*n_freq, B*C)
            spec = mx.matmul(basis, mx.transpose(seg, (1, 0)))
            spec = mx.transpose(spec, (1, 0))  # (B*C, 2*n_freq)
            real = spec[:, :n_freq]
            imag = spec[:, n_freq:]
            magnitude = mx.sqrt(real * real + imag * imag)
            mel = mx.matmul(magnitude, mx.transpose(mel_basis, (1, 0)))
            mel = mx.log(mx.maximum(mel, 1e-5))
            frames.append(mel)

        mel_bt = mx.stack(frames, axis=1)  # (B*C, T_frames, n_mels)
        mel = mx.reshape(mel_bt, (b, c, n_frames, mel_bt.shape[-1]))
        mel = mx.transpose(mel, (0, 1, 3, 2))  # (B, C, n_mels, T_frames)
        return mel

    def _upsample_skip(self, x: mx.array) -> mx.array:
        """Upsample skip connection to BWE sample rate via linear interpolation.

        Upstream uses a Hann-windowed sinc resampler. Linear interpolation is a
        reasonable MLX-portable approximation that avoids the spectral-image
        aliasing artefacts of nearest-neighbour repeat.
        """
        ratio = max(1, self.output_sampling_rate // self.input_sampling_rate)
        if ratio <= 1:
            return x
        # x: (B, C, T) — transpose to (B, T, C) for gather, then back
        x_btc = mx.transpose(x, (0, 2, 1))  # (B, T, C)
        b, t, c = x_btc.shape
        t_out = t * ratio
        idx = mx.arange(t_out, dtype=mx.float32) / ratio
        idx_floor = mx.clip(idx.astype(mx.int32), 0, t - 1)
        idx_ceil = mx.clip(idx_floor + 1, 0, t - 1)
        frac = (idx - idx_floor.astype(mx.float32)).reshape(1, t_out, 1)
        lo = x_btc[:, idx_floor, :]  # (B, T_out, C)
        hi = x_btc[:, idx_ceil, :]
        out = lo + frac * (hi - lo)
        return mx.transpose(out, (0, 2, 1))  # (B, C, T_out)

    def __call__(self, mel_spec: mx.array) -> mx.array:
        low = self.vocoder(mel_spec)  # (B, C, T_low)
        mel_from_low = self._compute_mel(low)  # (B, C, n_mels, T_frames)
        mel_for_bwe = mx.transpose(mel_from_low, (0, 1, 3, 2))  # (B, C, T, n_mels)
        residual = self.bwe_generator(mel_for_bwe)  # (B, C, T_high)
        skip = self._upsample_skip(low)  # (B, C, T_high_approx)

        target = min(residual.shape[2], skip.shape[2])
        residual = residual[:, :, :target]
        skip = skip[:, :, :target]
        pre_clip = residual + skip
        return mx.clip(pre_clip, -1.0, 1.0)
