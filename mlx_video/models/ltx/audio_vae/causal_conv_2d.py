"""Causal 2D convolutions for audio VAE."""

from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .causality_axis import CausalityAxis


def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert int or tuple to tuple pair."""
    if isinstance(x, int):
        return (x, x)
    return x


class CausalConv2d(nn.Module):
    """
    A causal 2D convolution.
    This layer ensures that the output at time `t` only depends on inputs
    at time `t` and earlier. It achieves this by applying asymmetric padding
    to the time dimension before the convolution.

    Note: MLX Conv2d expects input shape (N, H, W, C) - channels last.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()

        self.causality_axis = causality_axis

        # Ensure kernel_size and dilation are tuples
        kernel_size = _pair(kernel_size)
        dilation = _pair(dilation)

        # Calculate padding dimensions
        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]

        # Store padding for manual application
        # MLX pad order: [(before_axis0, after_axis0), (before_axis1, after_axis1), ...]
        # For (N, H, W, C) format: axis 1 is H (height), axis 2 is W (width)
        if self.causality_axis == CausalityAxis.NONE:
            # Non-causal: symmetric padding
            self.padding = (pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2)
        elif self.causality_axis in (CausalityAxis.WIDTH, CausalityAxis.WIDTH_COMPATIBILITY):
            # Causal on width: pad left (before width axis)
            self.padding = (pad_h // 2, pad_h - pad_h // 2, pad_w, 0)
        elif self.causality_axis == CausalityAxis.HEIGHT:
            # Causal on height: pad top (before height axis)
            self.padding = (pad_h, 0, pad_w // 2, pad_w - pad_w // 2)
        else:
            raise ValueError(f"Invalid causality_axis: {causality_axis}")

        # The internal convolution layer uses no padding, as we handle it manually
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with causal padding.
        Args:
            x: Input tensor of shape (N, H, W, C) in MLX channels-last format
        Returns:
            Output tensor after causal convolution
        """
        # Apply causal padding before convolution
        # padding format: (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)
        pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = self.padding

        if any(p > 0 for p in self.padding):
            # MLX pad expects: [(before_0, after_0), (before_1, after_1), ...]
            # For (N, H, W, C): axis 0=N, axis 1=H, axis 2=W, axis 3=C
            x = mx.pad(x, [(0, 0), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right), (0, 0)])

        return self.conv(x)


def make_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: int = 1,
    padding: Union[int, Tuple[int, int], None] = None,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causality_axis: CausalityAxis | None = None,
) -> nn.Module:
    """
    Create a 2D convolution layer that can be either causal or non-causal.
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Convolution stride
        padding: Padding (if None, will be calculated based on causal flag)
        dilation: Dilation rate
        groups: Number of groups for grouped convolution
        bias: Whether to use bias
        causality_axis: Dimension along which to apply causality.
    Returns:
        Either a regular Conv2d or CausalConv2d layer
    """
    if causality_axis is not None:
        # For causal convolution, padding is handled internally by CausalConv2d
        return CausalConv2d(
            in_channels, out_channels, kernel_size, stride, dilation, groups, bias, causality_axis
        )
    else:
        # For non-causal convolution, use symmetric padding if not specified
        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = tuple(k // 2 for k in kernel_size)

        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
