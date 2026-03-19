"""ResNet blocks for audio VAE and vocoder."""

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from .causal_conv_2d import make_conv2d
from .causality_axis import CausalityAxis
from .normalization import NormType, build_normalization_layer

LRELU_SLOPE = 0.1


def leaky_relu(x: mx.array, negative_slope: float = LRELU_SLOPE) -> mx.array:
    """Leaky ReLU activation."""
    return mx.maximum(x, x * negative_slope)


class _SnakeCore(nn.Module):
    """Core SnakeBeta activation with learnable alpha/beta (log-scale)."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,), dtype=mx.float32)
        self.beta = mx.zeros((channels,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        # x is (B, L, C) for MLX Conv1d.
        # Weights are stored in log-scale (BigVGAN convention with snake_logscale=True).
        alpha = mx.exp(mx.reshape(self.alpha, (1, 1, -1)))
        beta = mx.exp(mx.reshape(self.beta, (1, 1, -1)))
        return x + (mx.sin(alpha * x) ** 2) / (beta + 1e-6)


class _SnakeFilter(nn.Module):
    """Kaiser-sinc low-pass filter for anti-aliased activation (Activation1d).

    Checkpoint stores a (1, 1, taps) filter kernel.  At runtime we apply it as
    a depth-wise 1-D convolution along the time axis (MLX layout: B, T, C).
    """

    def __init__(self, taps: int = 12):
        super().__init__()
        self.filter = mx.zeros((1, 1, taps), dtype=mx.float32)

    def _apply_filter(self, x: mx.array, stride: int = 1) -> mx.array:
        """Depth-wise 1-D convolution with the stored kaiser-sinc kernel.

        Args:
            x: (B, T, C) — MLX channels-last layout.
            stride: temporal stride (1 for upsample path, 2 for downsample).
        Returns:
            (B, T_out, C)
        """
        filt = self.filter  # (1, 1, taps)
        taps = filt.shape[-1]
        even = taps % 2 == 0
        pad_left = taps // 2 - int(even)
        pad_right = taps // 2

        # x: (B, T, C) → (B, C, T) for grouped conv, then back
        b, t, c = x.shape
        x_bct = mx.transpose(x, (0, 2, 1))  # (B, C, T)

        # Replicate-pad along time axis
        left = mx.broadcast_to(x_bct[:, :, :1], (b, c, pad_left))
        right = mx.broadcast_to(x_bct[:, :, -1:], (b, c, pad_right))
        x_padded = mx.concatenate([left, x_bct, right], axis=2)  # (B, C, T+pad)

        # Expand filter to (C, 1, taps) for depth-wise conv
        filt_ckt = mx.broadcast_to(
            mx.reshape(filt, (1, 1, taps)), (c, 1, taps)
        )  # (C, 1, taps)

        # Depth-wise conv: treat each channel independently
        # MLX conv1d expects (B, T, C_in) input and (C_out, k, C_in) weight
        # For grouped/depth-wise we process each channel as a separate batch item
        x_flat = mx.reshape(x_padded, (b * c, 1, -1))  # (B*C, 1, T+pad)
        x_flat = mx.transpose(x_flat, (0, 2, 1))  # (B*C, T+pad, 1)
        w = mx.reshape(filt, (1, taps, 1))  # (1, taps, 1) — single-channel conv
        out = mx.conv1d(x_flat, w, stride=stride)  # (B*C, T_out, 1)
        out = mx.reshape(out, (b, c, -1))  # (B, C, T_out)
        return mx.transpose(out, (0, 2, 1))  # (B, T_out, C)


class _SnakeUpsample(_SnakeFilter):
    """2x upsample using zero-insert + conv_transpose with kaiser-sinc filter."""

    def __init__(self, taps: int = 12):
        super().__init__(taps)

    def __call__(self, x: mx.array) -> mx.array:
        """Upsample x by 2 using the stored filter.

        Args:
            x: (B, T, C)
        Returns:
            (B, T*2, C)
        """
        filt = self.filter  # (1, 1, taps)
        taps = filt.shape[-1]
        ratio = 2
        pad = taps // ratio - 1
        pad_left = pad * ratio + (taps - ratio) // 2
        pad_right = pad * ratio + (taps - ratio + 1) // 2

        b, t, c = x.shape
        x_bct = mx.transpose(x, (0, 2, 1))  # (B, C, T)

        # Replicate-pad
        left = mx.broadcast_to(x_bct[:, :, :1], (b, c, pad))
        right = mx.broadcast_to(x_bct[:, :, -1:], (b, c, pad))
        x_padded = mx.concatenate([left, x_bct, right], axis=2)  # (B, C, T+2*pad)

        # Transpose-conv per channel (zero-insert + filter)
        # Process each channel independently
        x_flat = mx.reshape(x_padded, (b * c, 1, -1))  # (B*C, 1, T_padded)
        x_flat = mx.transpose(x_flat, (0, 2, 1))  # (B*C, T_padded, 1)
        w = mx.reshape(filt, (1, taps, 1))  # (1, taps, 1)
        out = ratio * mx.conv_transpose1d(x_flat, w, stride=ratio)  # (B*C, T_up, 1)
        out = mx.reshape(out, (b, c, -1))  # (B, C, T_up)
        out = out[:, :, pad_left:]
        if pad_right > 0:
            out = out[:, :, :-pad_right]
        return mx.transpose(out, (0, 2, 1))  # (B, T_out, C)


class _SnakeDownsample(nn.Module):
    """2x downsample using low-pass filter + stride-2 convolution."""

    def __init__(self):
        super().__init__()
        self.lowpass = _SnakeFilter()

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass._apply_filter(x, stride=2)


class SnakeBeta(nn.Module):
    """BigVGAN-style SnakeBeta activation with anti-aliased Activation1d wrapper.

    Pipeline: upsample 2x → SnakeBeta → low-pass filter + downsample 2x.
    This prevents the sin²(αx) harmonics from aliasing back into the signal.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.act = _SnakeCore(channels)
        self.upsample = _SnakeUpsample()
        self.downsample = _SnakeDownsample()

    def __call__(self, x: mx.array) -> mx.array:
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


class ResBlock1(nn.Module):
    """1D ResNet block for vocoder with dilated convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()

        # First set of convolutions with different dilations
        self.convs1 = {
            i: nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=(kernel_size - 1) * d // 2,
            )
            for i, d in enumerate(dilation)
        }

        # Second set of convolutions with dilation=1
        self.convs2 = {
            i: nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=(kernel_size - 1) // 2,
            )
            for i in range(len(dilation))
        }

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through residual blocks."""
        for i in range(len(self.convs1)):
            xt = leaky_relu(x, LRELU_SLOPE)
            xt = self.convs1[i](xt)
            xt = leaky_relu(xt, LRELU_SLOPE)
            xt = self.convs2[i](xt)
            x = xt + x
        return x


class AMPBlock1(nn.Module):
    """BigVGAN AMP1 residual block using SnakeBeta activations."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()

        self.convs1 = {
            i: nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=(kernel_size - 1) * d // 2,
            )
            for i, d in enumerate(dilation)
        }
        self.convs2 = {
            i: nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=(kernel_size - 1) // 2,
            )
            for i in range(len(dilation))
        }
        self.acts1 = {i: SnakeBeta(channels) for i in range(len(dilation))}
        self.acts2 = {i: SnakeBeta(channels) for i in range(len(dilation))}

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(len(self.convs1)):
            xt = self.acts1[i](x)
            xt = self.convs1[i](xt)
            xt = self.acts2[i](xt)
            xt = self.convs2[i](xt)
            x = xt + x
        return x


class ResBlock2(nn.Module):
    """1D ResNet block for vocoder (alternative version)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int] = (1, 3),
    ):
        super().__init__()

        self.convs = {
            i: nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=(kernel_size - 1) * d // 2,
            )
            for i, d in enumerate(dilation)
        }

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through residual blocks."""
        for i in range(len(self.convs)):
            xt = leaky_relu(x, LRELU_SLOPE)
            xt = self.convs[i](xt)
            x = xt + x
        return x


class ResnetBlock(nn.Module):
    """2D ResNet block for audio VAE encoder/decoder."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and norm_type == NormType.GROUP:
            raise ValueError("Causal ResnetBlock with GroupNorm is not supported.")

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.temb_channels = temb_channels

        self.norm1 = build_normalization_layer(in_channels, normtype=norm_type)
        self.conv1 = make_conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = build_normalization_layer(out_channels, normtype=norm_type)
        self.dropout_rate = dropout
        self.conv2 = make_conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = make_conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    causality_axis=causality_axis,
                )
            else:
                self.nin_shortcut = make_conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    causality_axis=causality_axis,
                )

    def __call__(
        self,
        x: mx.array,
        temb: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass through ResNet block.
        Args:
            x: Input tensor of shape (N, H, W, C) in MLX channels-last format
            temb: Optional time embedding tensor
        Returns:
            Output tensor
        """
        h = x
        h = self.norm1(h)
        h = nn.silu(h)
        h = self.conv1(h)

        if temb is not None and self.temb_channels > 0:
            # temb: (B, temb_channels) -> (B, out_channels)
            # Need to add spatial dims: (B, 1, 1, out_channels) for broadcasting
            h = h + mx.expand_dims(
                mx.expand_dims(nn.silu(self.temb_proj(temb)), axis=1), axis=1
            )

        h = self.norm2(h)
        h = nn.silu(h)
        if self.dropout_rate > 0:
            h = nn.Dropout(self.dropout_rate)(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
