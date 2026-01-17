from typing import Tuple, Union
import mlx.core as mx
import mlx.nn as nn


class Conv3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Weight shape: (C_out, KD, KH, KW, C_in)
        scale = 1.0 / (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]) ** 0.5
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size[0], kernel_size[1], kernel_size[2], in_channels),
        )

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, D, H, W, C_in)

        Returns:
            Output tensor of shape (N, D', H', W', C_out)
        """
        y = mx.conv3d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y


class GroupNorm3d(nn.Module):

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, D, H, W, C)
        n, d, h, w, c = x.shape
        input_dtype = x.dtype


        x = x.astype(mx.float32)

        # Reshape to (N, D*H*W, num_groups, C//num_groups)
        x = mx.reshape(x, (n, d * h * w, self.num_groups, c // self.num_groups))

        # Compute mean and var over spatial and channel group dims
        mean = mx.mean(x, axis=(1, 3), keepdims=True)
        var = mx.var(x, axis=(1, 3), keepdims=True)

        # Normalize
        x = (x - mean) / mx.sqrt(var + self.eps)

        # Reshape back
        x = mx.reshape(x, (n, d, h, w, c))

        # Apply weight and bias
        weight = self.weight.astype(mx.float32)
        bias = self.bias.astype(mx.float32)
        x = x * weight + bias

        # Convert back to input dtype
        x = x.astype(input_dtype)

        return x


class PixelShuffle2D(nn.Module):
    """Pixel shuffle for 2D spatial upsampling."""

    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.upscale_factor = upscale_factor

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, H, W, C) where C = out_channels * upscale_factor^2
        n, h, w, c = x.shape
        r = self.upscale_factor
        out_c = c // (r * r)

        # Reshape: (N, H, W, out_c, r, r)
        x = mx.reshape(x, (n, h, w, out_c, r, r))

        # Permute: (N, H, r, W, r, out_c)
        x = mx.transpose(x, (0, 1, 4, 2, 5, 3))

        # Reshape: (N, H*r, W*r, out_c)
        x = mx.reshape(x, (n, h * r, w * r, out_c))

        return x


class SpatialRationalResampler(nn.Module):

    def __init__(self, mid_channels: int = 1024, scale: float = 2.0):
        super().__init__()
        self.scale = scale

        # 2D conv: mid_channels -> 4*mid_channels for pixel shuffle
        self.conv = nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1)

        # Blur kernel for antialiasing
        self.blur_down_kernel = mx.ones((1, 1, 5, 5)) / 25.0

        self.pixel_shuffle = PixelShuffle2D(2)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, D, H, W, C) - channels last 3D format

        n, d, h, w, c = x.shape

        # Process frame by frame
        # Reshape to (N*D, H, W, C) for 2D operations
        x = mx.reshape(x, (n * d, h, w, c))

        # Apply 2D conv
        x = self.conv(x)

        # Pixel shuffle for 2x upscaling
        x = self.pixel_shuffle(x)

        # Reshape back to (N, D, H*2, W*2, C)
        x = mx.reshape(x, (n, d, h * 2, w * 2, c))

        return x


class ResBlock3D(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = GroupNorm3d(32, channels)
        self.conv2 = Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = GroupNorm3d(32, channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.silu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        # Activation AFTER residual addition
        x = nn.silu(x + residual)

        return x


class LatentUpsampler(nn.Module):


    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 1024,
        num_blocks_per_stage: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels

        # Initial projection
        self.initial_conv = Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = GroupNorm3d(32, mid_channels)

        # Pre-upsample ResBlocks - use dict with int keys for MLX parameter tracking
        self.res_blocks = {i: ResBlock3D(mid_channels) for i in range(num_blocks_per_stage)}

        # Upsampler: 2D spatial upsampling (frame-by-frame)
        self.upsampler = SpatialRationalResampler(mid_channels=mid_channels, scale=2.0)

        # Post-upsample ResBlocks - use dict with int keys for MLX parameter tracking
        self.post_upsample_res_blocks = {i: ResBlock3D(mid_channels) for i in range(num_blocks_per_stage)}

        # Final projection
        self.final_conv = Conv3d(mid_channels, in_channels, kernel_size=3, padding=1)

    def __call__(self, latent: mx.array, debug: bool = False) -> mx.array:
        """Upsample latents by 2x spatially.

        Args:
            latent: Input tensor of shape (B, C, F, H, W) - channels first
            debug: If True, print intermediate values for debugging

        Returns:
            Upsampled tensor of shape (B, C, F, H*2, W*2) - channels first
        """
        def debug_stats(name, t):
            if debug:
                mx.eval(t)
                print(f"    {name}: shape={t.shape}, min={t.min().item():.4f}, max={t.max().item():.4f}, mean={t.mean().item():.4f}")

        if debug:
            print("  [DEBUG] LatentUpsampler forward pass:")
            debug_stats("Input (channels first)", latent)

        # Convert from channels first (B, C, F, H, W) to channels last (B, F, H, W, C)
        x = mx.transpose(latent, (0, 2, 3, 4, 1))
        if debug:
            debug_stats("After transpose to channels-last", x)

        # Initial conv
        x = self.initial_conv(x)
        if debug:
            debug_stats("After initial_conv", x)
        x = self.initial_norm(x)
        if debug:
            debug_stats("After initial_norm", x)
        x = nn.silu(x)
        if debug:
            debug_stats("After silu", x)

        # Pre-upsample blocks
        for i in sorted(self.res_blocks.keys()):
            x = self.res_blocks[i](x)
            if debug:
                debug_stats(f"After res_blocks[{i}]", x)

        # Upsample (2D spatial, frame-by-frame)
        x = self.upsampler(x)
        if debug:
            debug_stats("After upsampler (spatial 2x)", x)

        # Post-upsample blocks
        for i in sorted(self.post_upsample_res_blocks.keys()):
            x = self.post_upsample_res_blocks[i](x)
            if debug:
                debug_stats(f"After post_upsample_res_blocks[{i}]", x)

        # Final conv
        x = self.final_conv(x)
        if debug:
            debug_stats("After final_conv", x)

        # Convert back to channels first (B, C, F, H, W)
        x = mx.transpose(x, (0, 4, 1, 2, 3))
        if debug:
            debug_stats("Output (channels first)", x)

        return x


def upsample_latents(
    latent: mx.array,
    upsampler: LatentUpsampler,
    latent_mean: mx.array,
    latent_std: mx.array,
    debug: bool = False,
) -> mx.array:

    # Un-normalize: latent * std + mean
    latent_mean = latent_mean.reshape(1, -1, 1, 1, 1)
    latent_std = latent_std.reshape(1, -1, 1, 1, 1)
    latent = latent * latent_std + latent_mean
   
    # Upsample
    latent = upsampler(latent, debug=debug)
   
    # Re-normalize: (latent - mean) / std
    latent = (latent - latent_mean) / latent_std

    return latent


def load_upsampler(weights_path: str) -> LatentUpsampler:
    """Load upsampler from safetensors weights.

    Args:
        weights_path: Path to upsampler weights file

    Returns:
        Loaded LatentUpsampler model
    """
    print(f"Loading spatial upsampler from {weights_path}...")
    raw_weights = mx.load(weights_path)

    # Check weight shapes to determine mid_channels
    # res_blocks.0.conv1.weight should be (mid_channels, mid_channels, 3, 3, 3)
    sample_key = "res_blocks.0.conv1.weight"
    if sample_key in raw_weights:
        mid_channels = raw_weights[sample_key].shape[0]
    else:
        mid_channels = 1024  # default

    print(f"  Detected mid_channels: {mid_channels}")

    # Create model
    upsampler = LatentUpsampler(
        in_channels=128,
        mid_channels=mid_channels,
        num_blocks_per_stage=4,
    )

    # Sanitize weights - convert from PyTorch to MLX format
    sanitized = {}
    for key, value in raw_weights.items():
        new_key = key

        # Conv3d weights: PyTorch (O, I, D, H, W) -> MLX (O, D, H, W, I)
        if "conv" in key and "weight" in key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Conv2d weights: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
        if "conv" in key and "weight" in key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        # Map upsampler.conv to upsampler.conv (SpatialRationalResampler)
        # Keys: upsampler.conv.weight, upsampler.conv.bias, upsampler.blur_down.kernel
        if key.startswith("upsampler."):
            new_key = key  # Keep as is for SpatialRationalResampler

        sanitized[new_key] = value

    # Load weights
    upsampler.load_weights(list(sanitized.items()), strict=False)

    print(f"  Loaded {len(sanitized)} weights")

    return upsampler
