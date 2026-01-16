"""ResNet blocks for Video VAE."""

from enum import Enum
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_video.models.ltx.video_vae.convolution import CausalConv3d, PaddingModeType
from mlx_video.utils import PixelNorm


class NormLayerType(Enum):
    GROUP_NORM = "group_norm"
    PIXEL_NORM = "pixel_norm"


def get_norm_layer(
    norm_type: NormLayerType,
    num_channels: int,
    num_groups: int = 32,
    eps: float = 1e-6,
) -> nn.Module:

    if norm_type == NormLayerType.GROUP_NORM:
        return nn.GroupNorm(num_groups=num_groups, dims=num_channels, eps=eps)
    elif norm_type == NormLayerType.PIXEL_NORM:
        return PixelNorm(eps=eps)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


class ResnetBlock3D(nn.Module):

    def __init__(
        self,
        dims: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        eps: float = 1e-6,
        groups: int = 32,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        
        super().__init__()

        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inject_noise = inject_noise

        # First normalization and convolution
        self.norm1 = get_norm_layer(norm_layer, in_channels, groups, eps)
        self.conv1 = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        # Second normalization and convolution
        self.norm2 = get_norm_layer(norm_layer, out_channels, groups, eps)
        self.conv2 = CausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        # Shortcut connection if channels change
        if in_channels != out_channels:
            self.shortcut = CausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                spatial_padding_mode=spatial_padding_mode,
            )
        else:
            self.shortcut = None

        # Activation
        self.act = nn.SiLU()

    def __call__(
        self,
        x: mx.array,
        causal: bool = True,
        generator: Optional[int] = None,
    ) -> mx.array:
       
        residual = x

        # First block
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x, causal=causal)

        # Inject noise if enabled
        if self.inject_noise and generator is not None:
            noise = mx.random.normal(x.shape)
            x = x + noise * 0.01

        # Second block
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x, causal=causal)

        # Shortcut
        if self.shortcut is not None:
            residual = self.shortcut(residual, causal=causal)

        return x + residual


class UNetMidBlock3D(nn.Module):

    def __init__(
        self,
        dims: int,
        in_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        attention_head_dim: Optional[int] = None,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        
        super().__init__()

        self.num_layers = num_layers

        # Create ResNet blocks - use dict for MLX parameter tracking
        # Named res_blocks to match PyTorch weight keys
        self.res_blocks = {
            i: ResnetBlock3D(
                dims=dims,
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                norm_layer=norm_layer,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,
                spatial_padding_mode=spatial_padding_mode,
            )
            for i in range(num_layers)
        }

    def __call__(
        self,
        x: mx.array,
        causal: bool = True,
        timestep: Optional[mx.array] = None,
        generator: Optional[int] = None,
    ) -> mx.array:

        for resnet in self.res_blocks.values():
            x = resnet(x, causal=causal, generator=generator)

        return x
