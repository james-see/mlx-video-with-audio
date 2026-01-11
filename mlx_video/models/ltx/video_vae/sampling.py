"""Sampling operations for Video VAE (upsampling/downsampling)."""

from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_video.models.ltx.video_vae.convolution import CausalConv3d, PaddingModeType


class SpaceToDepthDownsample(nn.Module):
    def __init__(
        self,
        dims: int,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int, int]],
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        
        super().__init__()

        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.stride = stride
        self.dims = dims

        # Calculate the multiplier for channels
        multiplier = stride[0] * stride[1] * stride[2]
        intermediate_channels = in_channels * multiplier

        # 1x1x1 convolution to adjust channels
        self.conv = CausalConv3d(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            spatial_padding_mode=spatial_padding_mode,
        )

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        
        b, c, d, h, w = x.shape
        st, sh, sw = self.stride

        # Pad if necessary to make dimensions divisible by stride
        pad_d = (st - d % st) % st
        pad_h = (sh - h % sh) % sh
        pad_w = (sw - w % sw) % sw

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # For causal, pad at the end of temporal dimension
            if causal:
                x = mx.pad(x, [(0, 0), (0, 0), (0, pad_d), (0, pad_h), (0, pad_w)])
            else:
                x = mx.pad(x, [(0, 0), (0, 0), (pad_d // 2, pad_d - pad_d // 2),
                              (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)])

        b, c, d, h, w = x.shape

        # Reshape to group spatial elements
        # (B, C, D, H, W) -> (B, C, D/st, st, H/sh, sh, W/sw, sw)
        x = mx.reshape(x, (b, c, d // st, st, h // sh, sh, w // sw, sw))

        # Permute to move stride elements to channel dim
        # (B, C, D', st, H', sh, W', sw) -> (B, C, st, sh, sw, D', H', W')
        x = mx.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))

        # Reshape to combine channels
        # (B, C, st, sh, sw, D', H', W') -> (B, C*st*sh*sw, D', H', W')
        new_c = c * st * sh * sw
        new_d = d // st
        new_h = h // sh
        new_w = w // sw
        x = mx.reshape(x, (b, new_c, new_d, new_h, new_w))

        # Apply 1x1 conv to adjust channels
        x = self.conv(x, causal=causal)

        return x


class DepthToSpaceUpsample(nn.Module):
    
    def __init__(
        self,
        dims: int,
        in_channels: int,
        stride: Union[int, Tuple[int, int, int]],
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
       
        super().__init__()

        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.stride = stride
        self.dims = dims
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor

        # Calculate output channels
        multiplier = stride[0] * stride[1] * stride[2]
        out_channels = in_channels // out_channels_reduction_factor
        self.out_channels = out_channels

        # 3x3x3 convolution to prepare channels for unpacking (matches PyTorch)
        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels * multiplier,
            kernel_size=3,
            stride=1,
            padding=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def _depth_to_space(self, x: mx.array) -> mx.array:
        b, c_packed, d, h, w = x.shape
        st, sh, sw = self.stride
        c = c_packed // (st * sh * sw)

        # (B, C*st*sh*sw, D, H, W) -> (B, C, st, sh, sw, D, H, W)
        x = mx.reshape(x, (b, c, st, sh, sw, d, h, w))

        # (B, C, st, sh, sw, D, H, W) -> (B, C, D, st, H, sh, W, sw)
        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))

        # (B, C, D, st, H, sh, W, sw) -> (B, C, D*st, H*sh, W*sw)
        x = mx.reshape(x, (b, c, d * st, h * sh, w * sw))

        return x

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        
        b, c, d, h, w = x.shape
        st, sh, sw = self.stride

        # Compute residual path if enabled
        x_residual = None
        if self.residual:
            # Reshape input: treat channels as spatial factors
            # "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)"
            x_residual = self._depth_to_space(x)

            # Tile channels to match output (PyTorch .repeat() tiles, not element-repeat!)
            # num_repeat = prod(stride) / out_channels_reduction_factor
            num_repeat = (st * sh * sw) // self.out_channels_reduction_factor
            x_residual = mx.tile(x_residual, (1, num_repeat, 1, 1, 1))

            # Remove first temporal frame if temporal upsampling
            if st > 1:
                x_residual = x_residual[:, :, 1:, :, :]

        # Apply conv
        x = self.conv(x, causal=causal)

        # Depth to space rearrangement
        x = self._depth_to_space(x)

        # Remove first frame for causal temporal upsampling
        if st > 1:
            x = x[:, :, 1:, :, :]

        # Add residual
        if self.residual and x_residual is not None:
            x = x + x_residual

        return x
