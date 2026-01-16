"""Audio VAE encoder and decoder for LTX-2."""

from typing import Set, Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention import AttentionType, make_attn
from .causal_conv_2d import make_conv2d
from .causality_axis import CausalityAxis
from .downsample import build_downsampling_path
from .normalization import NormType, build_normalization_layer
from .ops import AudioLatentShape, AudioPatchifier, PerChannelStatistics
from .resnet import ResnetBlock
from .upsample import build_upsampling_path

LATENT_DOWNSAMPLE_FACTOR = 4


def build_mid_block(
    channels: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    add_attention: bool,
) -> dict:
    """Build the middle block with two ResNet blocks and optional attention."""
    mid = {}
    mid["block_1"] = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    mid["attn_1"] = (
        make_attn(channels, attn_type=attn_type, norm_type=norm_type) if add_attention else None
    )
    mid["block_2"] = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    return mid


def run_mid_block(mid: dict, features: mx.array) -> mx.array:
    """Run features through the middle block."""
    features = mid["block_1"](features, temb=None)
    if mid["attn_1"] is not None:
        features = mid["attn_1"](features)
    return mid["block_2"](features, temb=None)


class AudioDecoder(nn.Module):
    """
    Symmetric decoder that reconstructs audio spectrograms from latent features.
    The decoder mirrors the encoder structure with configurable channel multipliers,
    attention resolutions, and causal convolutions.
    """

    def __init__(
        self,
        *,
        ch: int = 128,
        out_ch: int = 2,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Set[int] = None,
        resolution: int = 256,
        z_channels: int = 8,
        norm_type: NormType = NormType.PIXEL,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
        dropout: float = 0.0,
        mid_block_add_attention: bool = True,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: int | None = None,
    ) -> None:
        """
        Initialize the AudioDecoder.
        Args:
            ch: Base number of feature channels
            out_ch: Number of output channels (2 for stereo)
            ch_mult: Multiplicative factors for channels at each resolution
            num_res_blocks: Number of residual blocks per resolution
            attn_resolutions: Resolutions at which to apply attention
            resolution: Input spatial resolution
            z_channels: Number of latent channels
            norm_type: Normalization type
            causality_axis: Axis for causal convolutions
            dropout: Dropout probability
            mid_block_add_attention: Whether to add attention in middle block
            sample_rate: Audio sample rate
            mel_hop_length: Hop length for mel spectrogram
            is_causal: Whether to use causal convolutions
            mel_bins: Number of mel frequency bins
        """
        super().__init__()

        if attn_resolutions is None:
            attn_resolutions = {8, 16, 32}

        # Internal behavioral defaults
        resamp_with_conv = True
        attn_type = AttentionType.VANILLA

        # Per-channel statistics for denormalizing latents
        # Uses ch (base channel count) to match the patchified latent dimension
        # Input latent shape: (B, z_channels, T, latent_mel_bins) = (B, 8, T, 16)
        # After patchify: (B, T, z_channels * latent_mel_bins) = (B, T, 128)
        # ch=128 matches this dimension, so use ch for per_channel_statistics
        self.per_channel_statistics = PerChannelStatistics(latent_channels=ch)
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins

        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.out_ch = out_ch
        self.give_pre_end = False
        self.tanh_out = False
        self.norm_type = norm_type
        self.z_channels = z_channels
        self.channel_multipliers = ch_mult
        self.attn_resolutions = attn_resolutions
        self.causality_axis = causality_axis
        self.attn_type = attn_type

        base_block_channels = ch * self.channel_multipliers[-1]
        base_resolution = resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, z_channels, base_resolution, base_resolution)

        self.conv_in = make_conv2d(
            z_channels, base_block_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
        )

        self.mid = build_mid_block(
            channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=mid_block_add_attention,
        )

        self.up, final_block_channels = build_upsampling_path(
            ch=ch,
            ch_mult=ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=attn_resolutions,
            resamp_with_conv=resamp_with_conv,
            initial_block_channels=base_block_channels,
        )

        self.norm_out = build_normalization_layer(final_block_channels, normtype=self.norm_type)
        self.conv_out = make_conv2d(
            final_block_channels, out_ch, kernel_size=3, stride=1, causality_axis=self.causality_axis
        )

    def __call__(self, sample: mx.array) -> mx.array:
        """
        Decode latent features back to audio spectrograms.
        Args:
            sample: Encoded latent representation of shape (B, H, W, C) in MLX format
                    or (B, C, H, W) in PyTorch format (will be transposed)
        Returns:
            Reconstructed audio spectrogram
        """
        # Handle input format - if channels are in dim 1, transpose to channels-last
        if sample.shape[1] == self.z_channels and sample.ndim == 4:
            # PyTorch format (B, C, H, W) -> MLX format (B, H, W, C)
            sample = mx.transpose(sample, (0, 2, 3, 1))

        sample, target_shape = self._denormalize_latents(sample)

        h = self.conv_in(sample)
        h = run_mid_block(self.mid, h)
        h = self._run_upsampling_path(h)
        h = self._finalize_output(h)

        return self._adjust_output_shape(h, target_shape)

    def _denormalize_latents(self, sample: mx.array) -> tuple[mx.array, AudioLatentShape]:
        """Denormalize latents using per-channel statistics."""
        # sample shape: (B, H, W, C) in MLX format
        latent_shape = AudioLatentShape(
            batch=sample.shape[0],
            channels=sample.shape[3],  # channels last
            frames=sample.shape[1],  # height = frames
            mel_bins=sample.shape[2],  # width = mel_bins
        )

        sample_patched = self.patchifier.patchify(sample)
        sample_denormalized = self.per_channel_statistics.un_normalize(sample_patched)
        sample = self.patchifier.unpatchify(sample_denormalized, latent_shape)

        target_frames = latent_shape.frames * LATENT_DOWNSAMPLE_FACTOR
        if self.causality_axis != CausalityAxis.NONE:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        target_shape = AudioLatentShape(
            batch=latent_shape.batch,
            channels=self.out_ch,
            frames=target_frames,
            mel_bins=self.mel_bins if self.mel_bins is not None else latent_shape.mel_bins,
        )

        return sample, target_shape

    def _adjust_output_shape(
        self,
        decoded_output: mx.array,
        target_shape: AudioLatentShape,
    ) -> mx.array:
        """
        Adjust output shape to match target dimensions for variable-length audio.
        Args:
            decoded_output: Tensor of shape (B, H, W, C) in MLX format
            target_shape: AudioLatentShape describing target dimensions
        Returns:
            Tensor adjusted to match target_shape exactly
        """
        # Current output shape: (batch, frames, mel_bins, channels) in MLX format
        _, current_time, current_freq, _ = decoded_output.shape
        target_channels = target_shape.channels
        target_time = target_shape.frames
        target_freq = target_shape.mel_bins

        # Step 1: Crop first to avoid exceeding target dimensions
        decoded_output = decoded_output[
            :, : min(current_time, target_time), : min(current_freq, target_freq), :target_channels
        ]

        # Step 2: Calculate padding needed for time and frequency dimensions
        time_padding_needed = target_time - decoded_output.shape[1]
        freq_padding_needed = target_freq - decoded_output.shape[2]

        # Step 3: Apply padding if needed
        if time_padding_needed > 0 or freq_padding_needed > 0:
            # MLX pad: [(before_0, after_0), ...]
            # For (B, H, W, C): H=time, W=freq
            padding = [
                (0, 0),  # batch
                (0, max(time_padding_needed, 0)),  # time
                (0, max(freq_padding_needed, 0)),  # freq
                (0, 0),  # channels
            ]
            decoded_output = mx.pad(decoded_output, padding)

        # Step 4: Final safety crop to ensure exact target shape
        decoded_output = decoded_output[:, :target_time, :target_freq, :target_channels]

        # Transpose back to PyTorch format (B, C, H, W) for vocoder compatibility
        decoded_output = mx.transpose(decoded_output, (0, 3, 1, 2))

        return decoded_output

    def _run_upsampling_path(self, h: mx.array) -> mx.array:
        """Run through upsampling path."""
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx in range(len(stage["block"])):
                h = stage["block"][block_idx](h, temb=None)
                if block_idx in stage["attn"]:
                    h = stage["attn"][block_idx](h)

            if level != 0 and "upsample" in stage:
                h = stage["upsample"](h)

        return h

    def _finalize_output(self, h: mx.array) -> mx.array:
        """Apply final normalization and convolution."""
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)
        return mx.tanh(h) if self.tanh_out else h


def decode_audio(latent: mx.array, audio_decoder: AudioDecoder, vocoder: "Vocoder") -> mx.array:
    """
    Decode an audio latent representation using the provided audio decoder and vocoder.
    Args:
        latent: Input audio latent tensor
        audio_decoder: Model to decode the latent to spectrogram
        vocoder: Model to convert spectrogram to audio waveform
    Returns:
        Decoded audio as a float tensor
    """
    decoded_audio = audio_decoder(latent)
    decoded_audio = vocoder(decoded_audio)
    # Remove batch dimension if present
    if decoded_audio.shape[0] == 1:
        decoded_audio = decoded_audio[0]
    return decoded_audio.astype(mx.float32)
