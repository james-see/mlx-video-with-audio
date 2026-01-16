"""Video VAE Encoder for LTX-2 Image-to-Video.

The encoder compresses input images/videos to latent representations.
Used for I2V (image-to-video) conditioning by encoding the input image
to latent space, which can then be used to condition video generation.
"""

from pathlib import Path
from typing import List, Tuple, Any, Optional
import json

import mlx.core as mx
import mlx.nn as nn

from mlx_video.models.ltx.video_vae.video_vae import VideoEncoder, LogVarianceType, NormLayerType, PaddingModeType


def load_vae_encoder(model_path: str) -> VideoEncoder:
    """Load VAE encoder from safetensors file.

    Args:
        model_path: Path to the model weights (safetensors file or directory)

    Returns:
        Loaded VideoEncoder instance
    """
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

    print(f"Loading VAE encoder from {weights_path}...")

    # Read config from safetensors metadata
    encoder_blocks = []
    norm_layer = NormLayerType.PIXEL_NORM
    latent_log_var = LogVarianceType.UNIFORM
    patch_size = 4

    try:
        with safe_open(str(weights_path), framework="numpy") as f:
            metadata = f.metadata()
            if metadata and "config" in metadata:
                configs = json.loads(metadata["config"])
                vae_config = configs.get("vae", {})

                # Parse encoder blocks
                raw_blocks = vae_config.get("encoder_blocks", [])
                for block in raw_blocks:
                    if isinstance(block, list) and len(block) == 2:
                        name, params = block
                        encoder_blocks.append((name, params))

                # Parse other config
                norm_str = vae_config.get("norm_layer", "pixel_norm")
                norm_layer = NormLayerType.PIXEL_NORM if norm_str == "pixel_norm" else NormLayerType.GROUP_NORM

                var_str = vae_config.get("latent_log_var", "uniform")
                if var_str == "uniform":
                    latent_log_var = LogVarianceType.UNIFORM
                elif var_str == "per_channel":
                    latent_log_var = LogVarianceType.PER_CHANNEL
                elif var_str == "constant":
                    latent_log_var = LogVarianceType.CONSTANT
                else:
                    latent_log_var = LogVarianceType.NONE

                patch_size = vae_config.get("patch_size", 4)

                print(f"  Loaded config: {len(encoder_blocks)} encoder blocks, norm={norm_str}, patch_size={patch_size}")
    except Exception as e:
        print(f"  Could not read config from metadata: {e}")
        # Use default config
        encoder_blocks = [
            ("res_x", {"num_layers": 4}),
            ("compress_space_res", {"multiplier": 2}),
            ("res_x", {"num_layers": 6}),
            ("compress_time_res", {"multiplier": 2}),
            ("res_x", {"num_layers": 6}),
            ("compress_all_res", {"multiplier": 2}),
            ("res_x", {"num_layers": 2}),
            ("compress_all_res", {"multiplier": 2}),
            ("res_x", {"num_layers": 2}),
        ]
        print(f"  Using default encoder config with {len(encoder_blocks)} blocks")

    # Create encoder
    encoder = VideoEncoder(
        convolution_dimensions=3,
        in_channels=3,
        out_channels=128,
        encoder_blocks=encoder_blocks,
        patch_size=patch_size,
        norm_layer=norm_layer,
        latent_log_var=latent_log_var,
        encoder_spatial_padding_mode=PaddingModeType.ZEROS,
    )

    # Load weights
    weights = mx.load(str(weights_path))

    # Determine prefix based on weight keys
    has_vae_prefix = any(k.startswith("vae.") for k in weights.keys())

    if has_vae_prefix:
        prefix = "vae.encoder."
        stats_prefix = "vae.per_channel_statistics."
    else:
        prefix = "encoder."
        stats_prefix = "per_channel_statistics."

    # Load per-channel statistics for normalization
    mean_key = f"{stats_prefix}mean-of-means"
    std_key = f"{stats_prefix}std-of-means"

    if mean_key in weights:
        encoder.per_channel_statistics.mean = weights[mean_key]
        print(f"  Loaded latent mean: shape {weights[mean_key].shape}")
    if std_key in weights:
        encoder.per_channel_statistics.std = weights[std_key]
        print(f"  Loaded latent std: shape {weights[std_key].shape}")

    # Build encoder weights dict with key remapping
    encoder_weights = {}
    for key, value in weights.items():
        if not key.startswith(prefix):
            continue

        # Remove prefix
        new_key = key[len(prefix):]

        # Handle Conv3d weight transpose: (O, I, D, H, W) -> (O, D, H, W, I)
        if ".weight" in key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        encoder_weights[new_key] = value

    print(f"  Found {len(encoder_weights)} encoder weights")

    # Load weights
    encoder.load_weights(list(encoder_weights.items()), strict=False)

    print("VAE encoder loaded successfully")
    return encoder


def encode_image(
    image: mx.array,
    encoder: VideoEncoder,
) -> mx.array:
    """Encode a single image to latent space.

    Args:
        image: Image tensor of shape (H, W, 3) in range [0, 1] or (B, H, W, 3)
        encoder: Loaded VAE encoder

    Returns:
        Latent tensor of shape (1, 128, 1, H//32, W//32)
    """
    # Add batch dimension if needed
    if image.ndim == 3:
        image = mx.expand_dims(image, axis=0)  # (1, H, W, 3)

    # Convert from (B, H, W, C) to (B, C, H, W)
    image = mx.transpose(image, (0, 3, 1, 2))  # (B, 3, H, W)

    # Normalize to [-1, 1]
    if image.max() > 1.0:
        image = image / 255.0
    image = image * 2.0 - 1.0

    # Add temporal dimension: (B, C, H, W) -> (B, C, 1, H, W)
    image = mx.expand_dims(image, axis=2)  # (B, 3, 1, H, W)

    # Encode
    latent = encoder(image)

    return latent
