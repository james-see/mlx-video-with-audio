"""Attention blocks for audio VAE."""

from enum import Enum

import mlx.core as mx
import mlx.nn as nn

from .normalization import NormType, build_normalization_layer


class AttentionType(Enum):
    """Enum for specifying the attention mechanism type."""

    VANILLA = "vanilla"
    LINEAR = "linear"
    NONE = "none"


class AttnBlock(nn.Module):
    """Self-attention block for audio VAE."""

    def __init__(
        self,
        in_channels: int,
        norm_type: NormType = NormType.GROUP,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = build_normalization_layer(in_channels, normtype=norm_type)
        # Using Conv2d with kernel_size=1 for Q, K, V projections
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through attention block.
        Args:
            x: Input tensor of shape (B, H, W, C) in MLX channels-last format
        Returns:
            Output tensor with attention applied (residual connection)
        """
        h_ = x
        h_ = self.norm(h_)

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        # x shape: (B, H, W, C)
        b, h, w, c = q.shape

        # Reshape for attention: (B, H*W, C)
        q = q.reshape(b, h * w, c)
        k = k.reshape(b, h * w, c)
        v = v.reshape(b, h * w, c)

        # Attention: Q @ K^T / sqrt(d)
        # q: (B, HW, C), k: (B, HW, C) -> k^T: (B, C, HW)
        # w_: (B, HW, HW)
        scale = float(c) ** (-0.5)
        w_ = mx.matmul(q, k.transpose(0, 2, 1)) * scale
        w_ = mx.softmax(w_, axis=-1)

        # Attend to values
        # w_: (B, HW, HW), v: (B, HW, C) -> h_: (B, HW, C)
        h_ = mx.matmul(w_, v)

        # Reshape back to spatial dims
        h_ = h_.reshape(b, h, w, c)

        h_ = self.proj_out(h_)

        return x + h_


class Identity(nn.Module):
    """Identity module that returns input unchanged."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


def make_attn(
    in_channels: int,
    attn_type: AttentionType = AttentionType.VANILLA,
    norm_type: NormType = NormType.GROUP,
) -> nn.Module:
    """
    Create an attention module based on type.
    Args:
        in_channels: Number of input channels
        attn_type: Type of attention mechanism
        norm_type: Type of normalization
    Returns:
        Attention module
    """
    if attn_type == AttentionType.VANILLA:
        return AttnBlock(in_channels, norm_type=norm_type)
    elif attn_type == AttentionType.NONE:
        return Identity()
    elif attn_type == AttentionType.LINEAR:
        raise NotImplementedError(f"Attention type {attn_type.value} is not supported yet.")
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")
