"""Audio VAE module for LTX-2 audio generation."""

from .attention import AttentionType, AttnBlock, make_attn
from .audio_vae import AudioDecoder, decode_audio
from .causal_conv_2d import CausalConv2d, make_conv2d
from .causality_axis import CausalityAxis
from .downsample import Downsample, build_downsampling_path
from .normalization import NormType, PixelNorm, build_normalization_layer
from .ops import AudioLatentShape, AudioPatchifier, PerChannelStatistics
from .resnet import LRELU_SLOPE, ResBlock1, ResBlock2, ResnetBlock
from .upsample import Upsample, build_upsampling_path
from .vocoder import Vocoder

__all__ = [
    # Main components
    "AudioDecoder",
    "Vocoder",
    "decode_audio",
    # Ops
    "AudioLatentShape",
    "AudioPatchifier",
    "PerChannelStatistics",
    # Building blocks
    "AttentionType",
    "AttnBlock",
    "make_attn",
    "CausalConv2d",
    "make_conv2d",
    "CausalityAxis",
    "Downsample",
    "build_downsampling_path",
    "NormType",
    "PixelNorm",
    "build_normalization_layer",
    "ResBlock1",
    "ResBlock2",
    "ResnetBlock",
    "LRELU_SLOPE",
    "Upsample",
    "build_upsampling_path",
]
