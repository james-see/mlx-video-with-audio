"""Video VAE Decoder for LTX-2 with timestep conditioning.

Architecture (from PyTorch weights):
- conv_in: 128 -> 1024
- up_blocks.0: 5 ResBlocks at 1024 (with timestep)
- up_blocks.1: Conv 1024 -> 4096, depth2space -> 512, upscale 2x
- up_blocks.2: 5 ResBlocks at 512 (with timestep)
- up_blocks.3: Conv 512 -> 2048, depth2space -> 256, upscale 2x
- up_blocks.4: 5 ResBlocks at 256 (with timestep)
- up_blocks.5: Conv 256 -> 1024, depth2space -> 128, upscale 2x
- up_blocks.6: 5 ResBlocks at 128 (with timestep)
- pixel_norm + timestep modulation (last_scale_shift_table)
- conv_out: 128 -> 48
- unpatchify: 48 -> 3 with patch_size=4
"""

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_video.models.ltx.video_vae.convolution import CausalConv3d, PaddingModeType
from mlx_video.models.ltx.video_vae.ops import unpatchify
from mlx_video.models.ltx.video_vae.sampling import DepthToSpaceUpsample
from mlx_video.models.ltx.video_vae.tiling import TilingConfig, decode_with_tiling


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0,
    scale: float = 1,
    max_period: int = 10000,
) -> mx.array:
    """Create sinusoidal timestep embeddings."""
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = mx.exp(exponent)
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
    emb = scale * emb

    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])

    return emb


class TimestepEmbedding(nn.Module):
    """MLP for timestep embedding."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
        self.act = nn.SiLU()

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class PixArtAlphaTimestepEmbedder(nn.Module):
    """Combined timestep embedding (sinusoidal + MLP)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim
        )

    def __call__(self, timestep: mx.array, hidden_dtype: mx.Dtype = mx.float32) -> mx.array:
        timesteps_proj = get_timestep_embedding(
            timestep,
            embedding_dim=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0
        )
        timesteps_emb = self.timestep_embedder(timesteps_proj.astype(hidden_dtype))
        return timesteps_emb


class ResnetBlock3DSimple(nn.Module):
    """ResNet block with optional timestep conditioning.

    Weight keys: conv1.conv, conv2.conv, scale_shift_table
    """

    def __init__(
        self,
        channels: int,
        spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        timestep_conditioning: bool = False,
    ):
        super().__init__()
        self.timestep_conditioning = timestep_conditioning

        # Nested conv structure to match PyTorch naming: conv1.conv.weight
        self.conv1 = self._make_conv_wrapper(channels, channels, spatial_padding_mode)
        self.conv2 = self._make_conv_wrapper(channels, channels, spatial_padding_mode)

        self.act = nn.SiLU()

        # Scale-shift table for timestep conditioning: [shift1, scale1, shift2, scale2]
        if timestep_conditioning:
            self.scale_shift_table = mx.zeros((4, channels))

    def _make_conv_wrapper(self, in_ch, out_ch, padding_mode):
        """Create a wrapper object with a 'conv' attribute to match PyTorch naming."""
        class ConvWrapper(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.conv = CausalConv3d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    spatial_padding_mode=padding_mode,
                )
            def __call__(self_inner, x, causal=False):
                return self_inner.conv(x, causal=causal)
        return ConvWrapper()

    def pixel_norm(self, x: mx.array, eps: float = 1e-8) -> mx.array:
        """Apply pixel normalization."""
        return x / mx.sqrt(mx.mean(x ** 2, axis=1, keepdims=True) + eps)

    def __call__(
        self,
        x: mx.array,
        causal: bool = False,
        timestep_embed: Optional[mx.array] = None,
    ) -> mx.array:
        residual = x
        batch_size = x.shape[0]

        # Block 1 with optional timestep conditioning
        x = self.pixel_norm(x)

        if self.timestep_conditioning and timestep_embed is not None:
            # scale_shift_table: (4, C), timestep_embed: (B, 4*C, 1, 1, 1)
            # Combine table with timestep embedding
            ada_values = self.scale_shift_table[None, :, :, None, None, None]  # (1, 4, C, 1, 1, 1)
            # Reshape timestep_embed from (B, 4*C, 1, 1, 1) to (B, 4, C, 1, 1, 1)
            channels = self.scale_shift_table.shape[1]
            ts_reshaped = timestep_embed.reshape(batch_size, 4, channels, 1, 1, 1)
            ada_values = ada_values + ts_reshaped

            shift1 = ada_values[:, 0]  # (B, C, 1, 1, 1)
            scale1 = ada_values[:, 1]
            shift2 = ada_values[:, 2]
            scale2 = ada_values[:, 3]

            x = x * (1 + scale1) + shift1

        x = self.act(x)
        x = self.conv1(x, causal=causal)

        # Block 2 with optional timestep conditioning
        x = self.pixel_norm(x)

        if self.timestep_conditioning and timestep_embed is not None:
            x = x * (1 + scale2) + shift2

        x = self.act(x)
        x = self.conv2(x, causal=causal)

        return x + residual


class ResBlockGroup(nn.Module):
    """Group of ResNet blocks with shared timestep embedding.

    PyTorch naming: res_blocks.0, res_blocks.1, etc.
    """

    def __init__(
        self,
        channels: int,
        num_layers: int = 5,
        spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        timestep_conditioning: bool = False,
    ):
        super().__init__()
        self.timestep_conditioning = timestep_conditioning

        # Time embedder for this block group: embed_dim = 4 * channels
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaTimestepEmbedder(
                embedding_dim=channels * 4
            )

        # Use dict with int keys for MLX to track parameters properly
        self.res_blocks = {
            i: ResnetBlock3DSimple(
                channels,
                spatial_padding_mode,
                timestep_conditioning=timestep_conditioning
            )
            for i in range(num_layers)
        }

    def __call__(
        self,
        x: mx.array,
        causal: bool = False,
        timestep: Optional[mx.array] = None,
    ) -> mx.array:
        timestep_embed = None

        if self.timestep_conditioning and timestep is not None:
            batch_size = x.shape[0]
            timestep_embed = self.time_embedder(
                timestep.flatten(),
                hidden_dtype=x.dtype
            )
            # Reshape to (B, 4*C, 1, 1, 1) for broadcasting
            timestep_embed = timestep_embed.reshape(batch_size, -1, 1, 1, 1)

        for res_block in self.res_blocks.values():
            x = res_block(x, causal=causal, timestep_embed=timestep_embed)
        return x


class LTX2VideoDecoder(nn.Module):
    """LTX-2 Video VAE Decoder with timestep conditioning.

    Architecture:
    - conv_in: 128 -> 1024
    - up_blocks.0: 5 ResBlocks at 1024 (with timestep)
    - up_blocks.1: Upsampler 1024 -> 512
    - up_blocks.2: 5 ResBlocks at 512 (with timestep)
    - up_blocks.3: Upsampler 512 -> 256
    - up_blocks.4: 5 ResBlocks at 256 (with timestep)
    - up_blocks.5: Upsampler 256 -> 128
    - up_blocks.6: 5 ResBlocks at 128 (with timestep)
    - conv_out: 128 -> 48 (3 * 4^2 for patch_size=4)
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        num_layers_per_block: int = 5,
        spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        timestep_conditioning: bool = True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.timestep_conditioning = timestep_conditioning

        # Decode parameters (configurable via constructor)
        self.decode_noise_scale = 0.025  # Set to 0.0 to disable noise
        self.decode_timestep = 0.05

        # Per-channel statistics for denormalization (loaded from weights)
        self.latents_mean = mx.zeros((in_channels,))
        self.latents_std = mx.ones((in_channels,))

        # Initial conv: 128 -> 1024
        class ConvInWrapper(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.conv = CausalConv3d(
                    in_channels=in_channels,
                    out_channels=1024,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    spatial_padding_mode=spatial_padding_mode,
                )
            def __call__(self_inner, x, causal=False):
                return self_inner.conv(x, causal=causal)
        self.conv_in = ConvInWrapper()

        # Up blocks: alternating ResBlockGroup and DepthToSpaceUpsample
        # Use dict with int keys for MLX to track parameters properly
        self.up_blocks = {
            0: ResBlockGroup(1024, num_layers_per_block, spatial_padding_mode, timestep_conditioning),
            1: DepthToSpaceUpsample(
                dims=3,
                in_channels=1024,
                stride=(2, 2, 2),
                residual=True,
                out_channels_reduction_factor=2,
                spatial_padding_mode=spatial_padding_mode,
            ),
            2: ResBlockGroup(512, num_layers_per_block, spatial_padding_mode, timestep_conditioning),
            3: DepthToSpaceUpsample(
                dims=3,
                in_channels=512,
                stride=(2, 2, 2),
                residual=True,
                out_channels_reduction_factor=2,
                spatial_padding_mode=spatial_padding_mode,
            ),
            4: ResBlockGroup(256, num_layers_per_block, spatial_padding_mode, timestep_conditioning),
            5: DepthToSpaceUpsample(
                dims=3,
                in_channels=256,
                stride=(2, 2, 2),
                residual=True,
                out_channels_reduction_factor=2,
                spatial_padding_mode=spatial_padding_mode,
            ),
            6: ResBlockGroup(128, num_layers_per_block, spatial_padding_mode, timestep_conditioning),
        }

        final_out_channels = out_channels * patch_size * patch_size
        class ConvOutWrapper(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.conv = CausalConv3d(
                    in_channels=128,
                    out_channels=final_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    spatial_padding_mode=spatial_padding_mode,
                )
            def __call__(self_inner, x, causal=False):
                return self_inner.conv(x, causal=causal)
        self.conv_out = ConvOutWrapper()

        self.act = nn.SiLU()

        if timestep_conditioning:
            self.timestep_scale_multiplier = mx.array(1000.0)
            self.last_time_embedder = PixArtAlphaTimestepEmbedder(
                embedding_dim=128 * 2  # 256, matches (2, 128) table
            )
            self.last_scale_shift_table = mx.zeros((2, 128))

    def denormalize(self, x: mx.array) -> mx.array:
        """Denormalize latents using per-channel statistics."""
        dtype = x.dtype
        # Cast to float32 for precision (statistics may be in bfloat16)
        mean = self.latents_mean.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        std = self.latents_std.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        return (x * std + mean).astype(dtype)

    def pixel_norm(self, x: mx.array, eps: float = 1e-8) -> mx.array:
        """Apply pixel normalization."""
        return x / mx.sqrt(mx.mean(x ** 2, axis=1, keepdims=True) + eps)

    def __call__(
        self,
        sample: mx.array,
        causal: bool = False,
        timestep: Optional[mx.array] = None,
        debug: bool = False,
    ) -> mx.array:
       
        def debug_stats(name, t):
            if debug:
                mx.eval(t)
                print(f"  [VAE] {name}: shape={t.shape}, min={t.min().item():.4f}, max={t.max().item():.4f}, mean={t.mean().item():.4f}")

        batch_size = sample.shape[0]

        if debug:
            debug_stats("Input", sample)

        # Add noise if timestep conditioning is enabled
        if self.timestep_conditioning:
            noise = mx.random.normal(sample.shape) * self.decode_noise_scale
            sample = noise + (1.0 - self.decode_noise_scale) * sample
            if debug:
                debug_stats("After noise", sample)

        if debug:
            print(f"  [VAE] Denorm stats - mean: [{self.latents_mean.min().item():.4f}, {self.latents_mean.max().item():.4f}], std: [{self.latents_std.min().item():.4f}, {self.latents_std.max().item():.4f}]")
        sample = self.denormalize(sample)
        if debug:
            debug_stats("After denormalize", sample)

        if timestep is None and self.timestep_conditioning:
            timestep = mx.full((batch_size,), self.decode_timestep)

        scaled_timestep = None
        if self.timestep_conditioning and timestep is not None:
            scaled_timestep = timestep * self.timestep_scale_multiplier

        x = self.conv_in(sample, causal=causal)
        if debug:
            debug_stats("After conv_in", x)

        for i, block in self.up_blocks.items():
            if isinstance(block, ResBlockGroup):
                x = block(x, causal=causal, timestep=scaled_timestep)
            else:
                x = block(x, causal=causal)
            if debug:
                block_type = type(block).__name__
                debug_stats(f"After up_blocks[{i}] ({block_type})", x)

        x = self.pixel_norm(x)
        if debug:
            debug_stats("After pixel_norm", x)

        if self.timestep_conditioning and scaled_timestep is not None:
            embedded_timestep = self.last_time_embedder(
                scaled_timestep.flatten(),
                hidden_dtype=x.dtype
            )
            embedded_timestep = embedded_timestep.reshape(batch_size, -1, 1, 1, 1)

            ada_values = self.last_scale_shift_table[None, :, :, None, None, None]  # (1, 2, 128, 1, 1, 1)
            ts_reshaped = embedded_timestep.reshape(batch_size, 2, 128, 1, 1, 1)
            ada_values = ada_values + ts_reshaped

            shift = ada_values[:, 0]  # (B, 128, 1, 1, 1)
            scale = ada_values[:, 1]

            x = x * (1 + scale) + shift
            if debug:
                debug_stats("After timestep modulation", x)

        x = self.act(x)
        if debug:
            debug_stats("After activation", x)

        x = self.conv_out(x, causal=causal)
        if debug:
            debug_stats("After conv_out", x)

        # Unpatchify: (B, 48, F', H', W') -> (B, 3, F, H*4, W*4)
        x = unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)
        if debug:
            debug_stats("After unpatchify", x)

        return x

    def decode_tiled(
        self,
        sample: mx.array,
        tiling_config: Optional[TilingConfig] = None,
        causal: bool = False,
        timestep: Optional[mx.array] = None,
        debug: bool = False,
    ) -> mx.array:
        """Decode latents using tiling to reduce memory usage.

        This method is useful for decoding large videos that would otherwise
        cause out-of-memory errors. It divides the latents into tiles,
        decodes each tile separately, and blends them together.

        Args:
            sample: Input latents of shape (B, C, F, H, W).
            tiling_config: Tiling configuration. If None, uses TilingConfig.default().
            causal: Whether to use causal convolutions.
            timestep: Optional timestep for conditioning.
            debug: Whether to print debug info.

        Returns:
            Decoded video of shape (B, 3, F*8, H*8, W*8).
        """
        if tiling_config is None:
            tiling_config = TilingConfig.default()

        # Check if tiling is actually needed
        _, _, f, h, w = sample.shape
        needs_spatial_tiling = False
        needs_temporal_tiling = False

        # Spatial scale is 32 (8x VAE upsample + 4x unpatchify)
        # Temporal scale is 8
        spatial_scale = 32
        temporal_scale = 8

        if tiling_config.spatial_config is not None:
            s_cfg = tiling_config.spatial_config
            tile_size_latent = s_cfg.tile_size_in_pixels // spatial_scale
            if h > tile_size_latent or w > tile_size_latent:
                needs_spatial_tiling = True

        if tiling_config.temporal_config is not None:
            t_cfg = tiling_config.temporal_config
            tile_size_latent = t_cfg.tile_size_in_frames // temporal_scale
            if f > tile_size_latent:
                needs_temporal_tiling = True

        if not needs_spatial_tiling and not needs_temporal_tiling:
            # No tiling needed, use regular decode
            if debug:
                print("[Tiling] Input fits within tile size, using regular decode")
            return self(sample, causal=causal, timestep=timestep, debug=debug)

        if debug:
            print(f"[Tiling] Using tiled decode (spatial={needs_spatial_tiling}, temporal={needs_temporal_tiling})")

        return decode_with_tiling(
            decoder_fn=self,
            latents=sample,
            tiling_config=tiling_config,
            spatial_scale=32,  # VAE spatial: 8x upsampling + 4x unpatchify = 32x
            temporal_scale=8,  # VAE temporal upsampling factor
            causal=causal,
            timestep=timestep,
            debug=debug,
        )


def load_vae_decoder(model_path: str, timestep_conditioning: Optional[bool] = None) -> LTX2VideoDecoder:
    from pathlib import Path
    import json
    from safetensors import safe_open

    model_path = Path(model_path)

    # Try to find the weights file
    if model_path.is_file() and model_path.suffix == ".safetensors":
        weights_path = model_path
    elif (model_path / "ltx-2-19b-distilled.safetensors").exists():
        weights_path = model_path / "ltx-2-19b-distilled.safetensors"
    elif (model_path / "vae" / "diffusion_pytorch_model.safetensors").exists():
        weights_path = model_path / "vae" / "diffusion_pytorch_model.safetensors"
    else:
        raise FileNotFoundError(f"VAE weights not found at {model_path}")

    print(f"Loading VAE decoder from {weights_path}...")

    # Read config from safetensors metadata to auto-detect timestep_conditioning
    if timestep_conditioning is None:
        try:
            with safe_open(str(weights_path), framework="numpy") as f:
                metadata = f.metadata()
                if metadata and "config" in metadata:
                    configs = json.loads(metadata["config"])
                    vae_config = configs.get("vae", {})
                    timestep_conditioning = vae_config.get("timestep_conditioning", False)
                    print(f"  Auto-detected timestep_conditioning={timestep_conditioning} from weights")
                else:
                    timestep_conditioning = False
        except Exception as e:
            print(f"  Could not read config from metadata: {e}, defaulting to timestep_conditioning=False")
            timestep_conditioning = False

    decoder = LTX2VideoDecoder(timestep_conditioning=timestep_conditioning)

    weights = mx.load(str(weights_path))

    # Determine prefix based on weight keys
    has_vae_prefix = any(k.startswith("vae.") for k in weights.keys())
    has_decoder_prefix = any(k.startswith("decoder.") for k in weights.keys())

    if has_vae_prefix:
        prefix = "vae.decoder."
        stats_prefix = "vae.per_channel_statistics."
    elif has_decoder_prefix:
        prefix = "decoder."
        stats_prefix = ""
    else:
        prefix = ""
        stats_prefix = ""

    # Load per-channel statistics for denormalization
    # Note: use std-of-means (not mean-of-stds) for proper denormalization
    mean_key = f"{stats_prefix}mean-of-means" if stats_prefix else "latents_mean"
    std_key = f"{stats_prefix}std-of-means" if stats_prefix else "latents_std"

    if mean_key in weights:
        decoder.latents_mean = weights[mean_key]
        print(f"  Loaded latent mean: shape {decoder.latents_mean.shape}")
    if std_key in weights:
        decoder.latents_std = weights[std_key]
        print(f"  Loaded latent std: shape {decoder.latents_std.shape}")

    # Build decoder weights dict with key remapping
    decoder_weights = {}
    for key, value in weights.items():
        if not key.startswith(prefix):
            continue

        # Remove prefix
        new_key = key[len(prefix):]

        # Handle Conv3d weight transpose: (O, I, D, H, W) -> (O, D, H, W, I)
        if ".conv.weight" in key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))
        if ".conv.bias" in key:
            pass  # bias doesn't need transpose

       
        if ".conv.weight" in new_key or ".conv.bias" in new_key:
            if ".conv.conv.weight" not in new_key and ".conv.conv.bias" not in new_key:
                new_key = new_key.replace(".conv.weight", ".conv.conv.weight")
                new_key = new_key.replace(".conv.bias", ".conv.conv.bias")

        decoder_weights[new_key] = value

    print(f"  Found {len(decoder_weights)} decoder weights")

    ts_keys = [k for k in decoder_weights.keys() if "scale_shift" in k or "time_embedder" in k or "timestep_scale" in k]
    print(f"  Found {len(ts_keys)} timestep conditioning weights")

    # Load weights
    decoder.load_weights(list(decoder_weights.items()), strict=False)

    print("VAE decoder loaded successfully")
    return decoder
