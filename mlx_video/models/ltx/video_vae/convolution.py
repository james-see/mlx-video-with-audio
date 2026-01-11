from enum import Enum
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


class PaddingModeType(Enum):
    ZEROS = "zeros"
    REFLECT = "reflect"


def reflect_pad_2d(x: mx.array, pad_h: int, pad_w: int) -> mx.array:
    """Apply reflect padding to spatial dimensions of a 5D tensor.

    Args:
        x: Input tensor of shape (B, D, H, W, C) - channels last
        pad_h: Padding for height dimension
        pad_w: Padding for width dimension

    Returns:
        Padded tensor
    """
    if pad_h == 0 and pad_w == 0:
        return x

    # Height padding (axis 2)
    if pad_h > 0:
        # Get reflection indices - exclude boundary
        top_pad = x[:, :, 1:pad_h+1, :, :][:, :, ::-1, :, :]  # Flip top portion
        bottom_pad = x[:, :, -pad_h-1:-1, :, :][:, :, ::-1, :, :]  # Flip bottom portion
        x = mx.concatenate([top_pad, x, bottom_pad], axis=2)

    # Width padding (axis 3)
    if pad_w > 0:
        left_pad = x[:, :, :, 1:pad_w+1, :][:, :, :, ::-1, :]  # Flip left portion
        right_pad = x[:, :, :, -pad_w-1:-1, :][:, :, :, ::-1, :]  # Flip right portion
        x = mx.concatenate([left_pad, x, right_pad], axis=3)

    return x


def make_conv_nd(
    dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...], str] = 0,
    causal: bool = False,
    spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
) -> nn.Module:
    
    if dims == 2:
        return CausalConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif dims == 3:
        return CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"Unsupported number of dimensions: {dims}")


class CausalConv3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int], str] = 0,
        causal: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()

        self.causal = causal
        self.spatial_padding_mode = spatial_padding_mode

        # Normalize kernel_size and stride to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride
        self.time_kernel_size = kernel_size[0]

        # Calculate spatial padding (temporal is handled separately via frame replication)
        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        self.spatial_padding = (height_pad, width_pad)

        # Create the base convolution (without padding, we'll handle it manually)
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # We handle padding manually
            bias=True,
        )

    def __call__(self, x: mx.array, causal: Optional[bool] = None) -> mx.array:
        
        use_causal = causal if causal is not None else self.causal

        # Apply temporal padding via frame replication 
        # Only apply if kernel_size > 1
        if self.time_kernel_size > 1:
            if use_causal:
                # Causal: replicate first frame kernel_size-1 times at the beginning
                first_frame_pad = mx.repeat(x[:, :, :1, :, :], self.time_kernel_size - 1, axis=2)
                x = mx.concatenate([first_frame_pad, x], axis=2)
            else:
                # Non-causal: replicate first frame at start, last frame at end
                pad_size = (self.time_kernel_size - 1) // 2
                if pad_size > 0:
                    first_frame_pad = mx.repeat(x[:, :, :1, :, :], pad_size, axis=2)
                    last_frame_pad = mx.repeat(x[:, :, -1:, :, :], pad_size, axis=2)
                    x = mx.concatenate([first_frame_pad, x, last_frame_pad], axis=2)

        # Transpose to channels last: (B, C, D, H, W) -> (B, D, H, W, C)
        x = mx.transpose(x, (0, 2, 3, 4, 1))

        # Apply spatial padding
        pad_h, pad_w = self.spatial_padding
        if pad_h > 0 or pad_w > 0:
            if self.spatial_padding_mode == PaddingModeType.REFLECT:
                # Use reflect padding for spatial dimensions
                x = reflect_pad_2d(x, pad_h, pad_w)
            else:
                # Use zero padding for spatial dimensions
                pad_width = [
                    (0, 0),  # Batch
                    (0, 0),  # D (temporal - already padded)
                    (pad_h, pad_h),  # H
                    (pad_w, pad_w),  # W
                    (0, 0),  # C
                ]
                x = mx.pad(x, pad_width)

        # Apply convolution with chunking for large tensors
        # Note: We choose to use chunking because MLX conv3d fails around 33 frames with 192x192 spatial
        x = self._chunked_conv3d(x)

        # Transpose back to channels first: (B, D, H, W, C) -> (B, C, D, H, W)
        x = mx.transpose(x, (0, 4, 1, 2, 3))

        return x

    def _chunked_conv3d(self, x: mx.array) -> mx.array:
        """Apply conv3d in temporal chunks to work around MLX bug with large tensors.

        Args:
            x: Input tensor of shape (B, D, H, W, C) in channels-last format

        Returns:
            Output tensor after conv3d
        """
        b, d, h, w, c = x.shape


        total_elements = d * h * w * c
        max_safe_elements = 30 * 192 * 192 * 128  # ~140M elements per chunk

        if total_elements <= max_safe_elements:
            return self.conv(x)

        elements_per_frame = h * w * c
        max_frames_per_chunk = max(1, max_safe_elements // elements_per_frame)
        chunk_size = min(max_frames_per_chunk, 24)  # Cap at 24 frames per chunk

        kernel_t = self.time_kernel_size

        overlap = kernel_t - 1

      
        expected_output_frames = d - overlap

        outputs = []
        out_idx = 0 

        # Process chunks
        in_start = 0
        while out_idx < expected_output_frames:
            remaining = expected_output_frames - out_idx
            out_frames_this_chunk = min(chunk_size, remaining)

            in_frames_needed = out_frames_this_chunk + overlap
            in_end = min(in_start + in_frames_needed, d)

            chunk = x[:, in_start:in_end, :, :, :]

            chunk_out = self.conv(chunk)
            mx.eval(chunk_out)

            outputs.append(chunk_out)

            out_idx += chunk_out.shape[1]
            in_start += chunk_out.shape[1]

        # Concatenate all chunks
        if len(outputs) == 1:
            return outputs[0]
        return mx.concatenate(outputs, axis=1)


class CausalConv2d(nn.Module):
    """2D convolution with optional causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        causal: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        """Initialize CausalConv2d."""
        super().__init__()

        self.causal = causal
        self.spatial_padding_mode = spatial_padding_mode

        # Normalize kernel_size and stride
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride

        # Calculate padding
        if isinstance(padding, str) and padding == "same":
            self.padding = (
                (kernel_size[0] - 1) // 2,
                (kernel_size[1] - 1) // 2,
            )
        elif isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=True,
        )

    def __call__(self, x: mx.array, causal: Optional[bool] = None) -> mx.array:
        """Forward pass."""
        # Transpose to channels last: (B, C, H, W) -> (B, H, W, C)
        x = mx.transpose(x, (0, 2, 3, 1))

        # Apply padding
        pad_h, pad_w = self.padding
        if pad_h != 0 or pad_w != 0:
            pad_width = [
                (0, 0),  # Batch
                (pad_h, pad_h),  # H
                (pad_w, pad_w),  # W
                (0, 0),  # C
            ]
            x = mx.pad(x, pad_width)

        x = self.conv(x)

        # Transpose back: (B, H, W, C) -> (B, C, H, W)
        x = mx.transpose(x, (0, 3, 1, 2))

        return x
