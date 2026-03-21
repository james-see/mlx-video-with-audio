import json
from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mx_utils
from huggingface_hub import snapshot_download

from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType
from mlx_video.models.ltx.ltx import LTXModel

LTX2_MAIN_FILES = [
    "ltx-2-19b-distilled.safetensors",
    "ltx-2-19b-dev.safetensors",
]
LTX23_MAIN_FILES = [
    "ltx-2.3-22b-distilled.safetensors",
    "ltx-2.3-22b-dev.safetensors",
]
MAIN_MODEL_FILES = LTX23_MAIN_FILES + LTX2_MAIN_FILES


def detect_ltx_version(model_path: Path, hf_path: Optional[str] = None) -> str:
    """Detect LTX model family version from path/repo hint.

    Returns:
        "2.3" for LTX-2.3 style checkpoints, otherwise "2.0".
    """
    if hf_path and "2.3" in hf_path.lower():
        return "2.3"
    for filename in LTX23_MAIN_FILES:
        if (model_path / filename).exists():
            return "2.3"
    cfg_path = model_path / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            if str(cfg.get("model_version", "")).startswith("2.3"):
                return "2.3"
        except (OSError, json.JSONDecodeError):
            pass
    return "2.0"


def find_main_weight_file(model_path: Path) -> Optional[Path]:
    """Return primary distilled/dev checkpoint path when present."""
    for filename in MAIN_MODEL_FILES:
        candidate = model_path / filename
        if candidate.exists():
            return candidate
    return None


def get_model_path(
    path_or_hf_repo: str,
    revision: Optional[str] = None,
    allow_patterns: Optional[list[str]] = None,
) -> Path:
    """Get local path to model, downloading if necessary.

    Args:
        path_or_hf_repo: Local path or HuggingFace repo ID
        revision: Git revision for HF repo

    Returns:
        Path to model directory
    """
    model_path = Path(path_or_hf_repo).expanduser()

    if model_path.exists():
        return model_path

    patterns = allow_patterns or [
        "*.safetensors",
        "*.json",
        "config.json",
    ]

    # Download from HuggingFace
    model_path = Path(
        snapshot_download(
            repo_id=path_or_hf_repo,
            revision=revision,
            allow_patterns=patterns,
        )
    )

    return model_path


def load_safetensors(path: Path) -> Dict[str, mx.array]:
    """Load weights from safetensors file(s) using MLX.

    Args:
        path: Path to model directory or single safetensors file

    Returns:
        Dictionary of weights
    """
    weights = {}

    if path.is_file():
        # Single file - use mx.load directly (handles bfloat16)
        return mx.load(str(path))
    else:
        # Directory - load only the main model file (not quantized versions)
        # Priority: distilled > dev for known LTX2/LTX2.3 names.
        main_file = find_main_weight_file(path)
        if main_file is not None:
            print(f"  Loading from {main_file.name}")
            return mx.load(str(main_file))

        # Fallback: load first non-quantized safetensors file
        safetensor_files = list(path.glob("*.safetensors"))
        for sf_path in safetensor_files:
            # Skip quantized and lora variants
            if any(x in sf_path.name for x in ["-fp4", "-fp8", "-lora"]):
                continue
            print(f"  Loading from {sf_path.name}")
            return mx.load(str(sf_path))

    return weights


def load_transformer_weights(model_path: Path) -> Dict[str, mx.array]:
    """Load transformer weights from LTX-2 model.

    Args:
        model_path: Path to LTX-2 model directory

    Returns:
        Dictionary of transformer weights
    """
    # Try distilled model first, then dev for known LTX families.
    weight_file = find_main_weight_file(model_path)
    if weight_file is not None:
        print(f"Loading transformer weights from {weight_file.name}...")
        return mx.load(str(weight_file))

    raise FileNotFoundError(f"No transformer weights found in {model_path}")


def load_vae_weights(model_path: Path) -> Dict[str, mx.array]:
    """Load VAE weights from LTX-2 model.

    Args:
        model_path: Path to LTX-2 model directory

    Returns:
        Dictionary of VAE weights
    """
    vae_path = model_path / "vae" / "diffusion_pytorch_model.safetensors"
    if vae_path.exists():
        print(f"Loading VAE weights from {vae_path}...")
        return mx.load(str(vae_path))

    raise FileNotFoundError(f"VAE weights not found at {vae_path}")


def load_audio_vae_weights(model_path: Path) -> Dict[str, mx.array]:
    """Load audio VAE weights from LTX-2 model.

    Args:
        model_path: Path to LTX-2 model directory

    Returns:
        Dictionary of audio VAE weights
    """
    # Try different possible paths for audio VAE weights
    audio_vae_paths = [
        model_path / "audio_vae" / "diffusion_pytorch_model.safetensors",
        model_path / "audio_vae.safetensors",
    ]

    # Also check in main model weights
    main_paths = [model_path / name for name in MAIN_MODEL_FILES]

    for audio_path in audio_vae_paths:
        if audio_path.exists():
            print(f"Loading audio VAE weights from {audio_path}...")
            return mx.load(str(audio_path))

    # Check main model weights for audio_vae keys
    for main_path in main_paths:
        if main_path.exists():
            print(f"Loading audio VAE weights from {main_path.name}...")
            all_weights = mx.load(str(main_path))
            # Filter to only audio_vae keys
            audio_weights = {k: v for k, v in all_weights.items() if "audio_vae" in k}
            if audio_weights:
                return audio_weights

    raise FileNotFoundError(f"Audio VAE weights not found in {model_path}")


def load_vocoder_weights(model_path: Path) -> Dict[str, mx.array]:
    """Load vocoder weights from LTX-2 model.

    Args:
        model_path: Path to LTX-2 model directory

    Returns:
        Dictionary of vocoder weights
    """
    # Try different possible paths for vocoder weights
    vocoder_paths = [
        model_path / "vocoder" / "diffusion_pytorch_model.safetensors",
        model_path / "vocoder.safetensors",
    ]

    # Also check in main model weights
    main_paths = [model_path / name for name in MAIN_MODEL_FILES]

    for vocoder_path in vocoder_paths:
        if vocoder_path.exists():
            print(f"Loading vocoder weights from {vocoder_path}...")
            return mx.load(str(vocoder_path))

    # Check main model weights for vocoder keys
    for main_path in main_paths:
        if main_path.exists():
            print(f"Loading vocoder weights from {main_path.name}...")
            all_weights = mx.load(str(main_path))
            # Filter to only vocoder keys
            vocoder_weights = {k: v for k, v in all_weights.items() if "vocoder" in k}
            if vocoder_weights:
                return vocoder_weights

    raise FileNotFoundError(f"Vocoder weights not found in {model_path}")


def sanitize_transformer_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize transformer weight names from PyTorch LTX-2 format to MLX format.

    Args:
        weights: Dictionary of weights with PyTorch naming

    Returns:
        Dictionary with MLX-compatible naming for transformer
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # Skip non-transformer weights (VAE, vocoder, audio_vae, connectors)
        if not key.startswith("model.diffusion_model."):
            continue

        # Remove 'model.diffusion_model.' prefix
        new_key = key.replace("model.diffusion_model.", "")

        # Handle to_out.0 -> to_out (MLX doesn't use Sequential numbering)
        new_key = new_key.replace(".to_out.0.", ".to_out.")

        # Handle feed-forward net naming
        # PyTorch: ff.net.0.proj -> ff.net_0_proj (or similar)
        # MLX FeedForward: uses proj_in, proj_out
        new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
        new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
        new_key = new_key.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
        new_key = new_key.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")

        # Handle AdaLN naming - keep emb wrapper, just fix linear naming
        # PyTorch: adaln_single.emb.timestep_embedder.linear_1 -> adaln_single.emb.timestep_embedder.linear1
        new_key = new_key.replace(".linear_1.", ".linear1.")
        new_key = new_key.replace(".linear_2.", ".linear2.")

        # Handle caption projection (keep linear1/linear2 naming for compatibility)
        # These are already mapped correctly in the sanitization

        sanitized[new_key] = value

    return sanitized


def sanitize_vae_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize VAE weight names from PyTorch format to MLX format.

    Args:
        weights: Dictionary of weights with PyTorch naming

    Returns:
        Dictionary with MLX-compatible naming for VAE decoder
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # Skip position_ids (not needed)
        if "position_ids" in key:
            continue

        # Only process VAE decoder weights (skip audio_vae, etc.)
        if not key.startswith("vae."):
            continue

        # Handle per-channel statistics key mapping
        # PyTorch: vae.per_channel_statistics.mean-of-means -> per_channel_statistics.mean
        # PyTorch: vae.per_channel_statistics.std-of-means -> per_channel_statistics.std
        # Be careful: mean-of-stds_over_std-of-means also ends with std-of-means
        if "vae.per_channel_statistics" in key:
            if key == "vae.per_channel_statistics.mean-of-means":
                new_key = "per_channel_statistics.mean"
            elif key == "vae.per_channel_statistics.std-of-means":
                new_key = "per_channel_statistics.std"
            else:
                # Skip other per_channel_statistics keys (channel, mean-of-stds, etc.)
                continue
        elif key.startswith("vae.decoder."):
            # Strip the vae.decoder. prefix for decoder weights
            new_key = key.replace("vae.decoder.", "")
        else:
            # Skip other vae.* keys that are not decoder weights
            continue

        # Handle Conv3d weight shape conversion
        # PyTorch: (out_channels, in_channels, D, H, W)
        # MLX: (out_channels, D, H, W, in_channels)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 5:
            # Transpose from (O, I, D, H, W) to (O, D, H, W, I)
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Handle Conv2d weight shape conversion
        # PyTorch: (out_channels, in_channels, H, W)
        # MLX: (out_channels, H, W, in_channels)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value

    return sanitized


def sanitize_vae_encoder_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize VAE encoder weight names from PyTorch format to MLX format.

    Args:
        weights: Dictionary of weights with PyTorch naming

    Returns:
        Dictionary with MLX-compatible naming for VAE encoder
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # Skip position_ids (not needed)
        if "position_ids" in key:
            continue

        # Only process VAE encoder weights
        if not key.startswith("vae."):
            continue

        # Handle per-channel statistics key mapping
        if "vae.per_channel_statistics" in key:
            if key == "vae.per_channel_statistics.mean-of-means":
                new_key = "per_channel_statistics._mean_of_means"
            elif key == "vae.per_channel_statistics.std-of-means":
                new_key = "per_channel_statistics._std_of_means"
            else:
                # Skip other per_channel_statistics keys
                continue
        elif key.startswith("vae.encoder."):
            # Strip the vae.encoder. prefix for encoder weights
            new_key = key.replace("vae.encoder.", "")
        else:
            # Skip other vae.* keys that are not encoder weights
            continue

        # Handle Conv3d weight shape conversion
        # PyTorch: (out_channels, in_channels, D, H, W)
        # MLX: (out_channels, D, H, W, in_channels)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Handle Conv2d weight shape conversion
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value

    return sanitized


def sanitize_audio_vae_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize audio VAE weight names from PyTorch format to MLX format.

    Args:
        weights: Dictionary of weights with PyTorch naming

    Returns:
        Dictionary with MLX-compatible naming for audio VAE decoder
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # Handle audio_vae.decoder weights
        if key.startswith("audio_vae.decoder."):
            new_key = key.replace("audio_vae.decoder.", "")
        elif key.startswith("audio_vae.per_channel_statistics."):
            # Map per-channel statistics
            if "mean-of-means" in key:
                new_key = "per_channel_statistics._mean_of_means"
            elif "std-of-means" in key:
                new_key = "per_channel_statistics._std_of_means"
            else:
                continue  # Skip other statistics keys
        else:
            continue  # Skip non-decoder keys

        # Handle Conv2d weight shape conversion
        # PyTorch: (out_channels, in_channels, H, W)
        # MLX: (out_channels, H, W, in_channels)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value

    return sanitized


def sanitize_vocoder_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize vocoder weight names from PyTorch format to MLX format.

    Args:
        weights: Dictionary of weights with PyTorch naming

    Returns:
        Dictionary with MLX-compatible naming for vocoder
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # Handle vocoder weights
        if key.startswith("vocoder."):
            new_key = key.replace("vocoder.", "")

            # Handle ModuleList indices -> dict keys
            # PyTorch: ups.0, ups.1, ... -> ups.0, ups.1, ...
            # PyTorch: resblocks.0, resblocks.1, ... -> resblocks.0, resblocks.1, ...

            # Handle Conv1d weight shape conversion
            # PyTorch: (out_channels, in_channels, kernel)
            # MLX: (out_channels, kernel, in_channels)
            if "weight" in new_key and value.ndim == 3:
                if "ups" in new_key:
                    # ConvTranspose1d: PyTorch (in_ch, out_ch, kernel) -> MLX (out_ch, kernel, in_ch)
                    value = mx.transpose(value, (1, 2, 0))
                else:
                    # Conv1d: PyTorch (out_ch, in_ch, kernel) -> MLX (out_ch, kernel, in_ch)
                    value = mx.transpose(value, (0, 2, 1))

            sanitized[new_key] = value

    return sanitized


def sanitize_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize weight names from PyTorch format to MLX format.

    Generic function that handles both transformer and VAE weights.

    Args:
        weights: Dictionary of weights with PyTorch naming

    Returns:
        Dictionary with MLX-compatible naming
    """
    sanitized = {}

    for key, value in weights.items():
        new_key = key

        # Skip position_ids (not needed)
        if "position_ids" in key:
            continue

        # Handle transformer weights
        if key.startswith("model.diffusion_model."):
            new_key = key.replace("model.diffusion_model.", "")
            new_key = new_key.replace(".to_out.0.", ".to_out.")
            new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
            new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
            new_key = new_key.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
            new_key = new_key.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
            new_key = new_key.replace(".linear_1.", ".linear1.")
            new_key = new_key.replace(".linear_2.", ".linear2.")

        # Handle Conv3d weight shape conversion
        # PyTorch: (out_channels, in_channels, D, H, W)
        # MLX: (out_channels, D, H, W, in_channels)
        if "conv" in key.lower() and "weight" in key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Handle Conv2d weight shape conversion
        # PyTorch: (out_channels, in_channels, H, W)
        # MLX: (out_channels, H, W, in_channels)
        if "conv" in key.lower() and "weight" in key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value

    return sanitized


def load_config(model_path: Path) -> Dict[str, Any]:
    """Load model configuration.

    Args:
        model_path: Path to model directory

    Returns:
        Configuration dictionary
    """
    config_path = model_path / "config.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)

    # Return default config
    return {}


def create_model_from_config(config: Dict[str, Any]) -> LTXModel:
    """Create model instance from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        LTXModel instance
    """
    # Map config to LTXModelConfig
    caption_channels = config.get("caption_channels", 3840)
    if caption_channels is None:
        conn_heads = int(config.get("connector_num_attention_heads", 32))
        conn_head_dim = int(config.get("connector_attention_head_dim", 128))
        caption_channels = conn_heads * conn_head_dim

    audio_caption_channels = config.get("audio_caption_channels", 3840)
    if audio_caption_channels is None:
        audio_conn_heads = int(config.get("audio_connector_num_attention_heads", 32))
        audio_conn_head_dim = int(config.get("audio_connector_attention_head_dim", 64))
        audio_caption_channels = audio_conn_heads * audio_conn_head_dim

    apply_gated_attention = bool(config.get("apply_gated_attention", False))
    model_config = LTXModelConfig(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=config.get("num_attention_heads", 32),
        attention_head_dim=config.get("attention_head_dim", 128),
        in_channels=config.get("in_channels", 128),
        out_channels=config.get("out_channels", 128),
        num_layers=config.get("num_layers", 48),
        cross_attention_dim=config.get("cross_attention_dim", 4096),
        caption_channels=caption_channels,
        caption_projection_first_linear=config.get(
            "caption_projection_first_linear", True
        ),
        caption_projection_second_linear=config.get(
            "caption_projection_second_linear", True
        ),
        adaln_embedding_coefficient=(9 if apply_gated_attention else 6),
        apply_gated_attention=apply_gated_attention,
        audio_num_attention_heads=config.get("audio_num_attention_heads", 32),
        audio_attention_head_dim=config.get("audio_attention_head_dim", 64),
        audio_in_channels=config.get("audio_in_channels", 128),
        audio_out_channels=config.get("audio_out_channels", 128),
        audio_cross_attention_dim=config.get("audio_cross_attention_dim", 2048),
        audio_caption_channels=audio_caption_channels,
        positional_embedding_theta=config.get("positional_embedding_theta", 10000.0),
        positional_embedding_max_pos=config.get(
            "positional_embedding_max_pos", [20, 2048, 2048]
        ),
        audio_positional_embedding_max_pos=config.get(
            "audio_positional_embedding_max_pos", [20]
        ),
        timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
        av_ca_timestep_scale_multiplier=config.get(
            "av_ca_timestep_scale_multiplier", 1000
        ),
        norm_eps=config.get("norm_eps", 1e-6),
    )

    return LTXModel(model_config)


def build_output_config(version: str, include_audio: bool) -> Dict[str, Any]:
    """Create output config.json for unified/split MLX outputs."""
    config = {
        "model_type": "AudioVideo" if include_audio else "VideoOnly",
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "in_channels": 128,
        "out_channels": 128,
        "num_layers": 48,
        "cross_attention_dim": 4096,
        "caption_channels": 3840,
        "audio_num_attention_heads": 32,
        "audio_attention_head_dim": 64,
        "audio_in_channels": 128,
        "audio_out_channels": 128,
        "audio_cross_attention_dim": 2048,
        "positional_embedding_theta": 10000.0,
        "positional_embedding_max_pos": [20, 2048, 2048],
        "audio_positional_embedding_max_pos": [20],
        "timestep_scale_multiplier": 1000,
        "av_ca_timestep_scale_multiplier": 1000,
        "norm_eps": 1e-6,
        "audio_sample_rate": 24000,
        "audio_latent_sample_rate": 16000,
        "audio_hop_length": 160,
        "audio_latent_channels": 8,
        "audio_mel_bins": 16,
    }
    if version == "2.3":
        config.update(
            {
                "model_version": "2.3.0",
                "is_v2": True,
                "caption_channels": None,
                "apply_gated_attention": True,
                "cross_attention_adaln": True,
                "connector_positional_embedding_max_pos": [4096],
                "connector_rope_type": "SPLIT",
                "connector_num_attention_heads": 32,
                "connector_attention_head_dim": 128,
                "audio_connector_num_attention_heads": 32,
                "audio_connector_attention_head_dim": 64,
                "caption_projection_first_linear": False,
                "caption_projection_second_linear": False,
            }
        )
    return config


def build_embedded_config(version: str) -> Dict[str, Any]:
    """Create embedded_config.json used by runtime for advanced model features."""
    if version != "2.3":
        return {}
    return {
        "transformer": {
            "_class_name": "AVTransformer3DModel",
            "attention_head_dim": 128,
            "attention_type": "default",
            "caption_channels": 3840,
            "cross_attention_dim": 4096,
            "in_channels": 128,
            "norm_eps": 1e-6,
            "num_attention_heads": 32,
            "num_layers": 48,
            "out_channels": 128,
            "audio_num_attention_heads": 32,
            "audio_attention_head_dim": 64,
            "audio_out_channels": 128,
            "audio_cross_attention_dim": 2048,
            "audio_positional_embedding_max_pos": [20],
            "use_embeddings_connector": True,
            "connector_attention_head_dim": 128,
            "connector_num_attention_heads": 32,
            "connector_num_layers": 8,
            "connector_positional_embedding_max_pos": [4096],
            "connector_num_learnable_registers": 128,
            "use_middle_indices_grid": True,
            "apply_gated_attention": True,
            "connector_apply_gated_attention": True,
            "caption_projection_first_linear": False,
            "caption_projection_second_linear": False,
            "audio_connector_attention_head_dim": 64,
            "audio_connector_num_attention_heads": 32,
            "cross_attention_adaln": True,
            "text_encoder_norm_type": "per_token_rms",
            "rope_type": "split",
            "frequencies_precision": "float64",
            "positional_embedding_theta": 10000.0,
            "positional_embedding_max_pos": [20, 2048, 2048],
            "timestep_scale_multiplier": 1000,
            "av_ca_timestep_scale_multiplier": 1000.0,
        },
        "vae": {
            "_class_name": "CausalVideoAutoencoder",
            "dims": 3,
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 128,
            "patch_size": 4,
            "norm_layer": "pixel_norm",
            "spatial_padding_mode": "zeros",
            "timestep_conditioning": False,
            "decoder_blocks": [
                ["res_x", {"num_layers": 4}],
                ["compress_space", {"multiplier": 2}],
                ["res_x", {"num_layers": 6}],
                ["compress_time", {"multiplier": 2}],
                ["res_x", {"num_layers": 4}],
                ["compress_all", {"multiplier": 1}],
                ["res_x", {"num_layers": 2}],
                ["compress_all", {"multiplier": 2}],
                ["res_x", {"num_layers": 2}],
            ],
            "encoder_blocks": [
                ["res_x", {"num_layers": 4}],
                ["compress_space_res", {"multiplier": 2}],
                ["res_x", {"num_layers": 6}],
                ["compress_time_res", {"multiplier": 2}],
                ["res_x", {"num_layers": 4}],
                ["compress_all_res", {"multiplier": 2}],
                ["res_x", {"num_layers": 2}],
                ["compress_all_res", {"multiplier": 1}],
                ["res_x", {"num_layers": 2}],
            ],
        },
        "audio_vae": {
            "model": {
                "params": {
                    "ddconfig": {
                        "double_z": True,
                        "mel_bins": 64,
                        "z_channels": 8,
                        "resolution": 256,
                        "in_channels": 2,
                        "out_ch": 2,
                        "ch": 128,
                        "ch_mult": [1, 2, 4],
                        "num_res_blocks": 2,
                        "dropout": 0.0,
                        "mid_block_add_attention": False,
                        "norm_type": "pixel",
                        "causality_axis": "height",
                    },
                    "sampling_rate": 16000,
                }
            }
        },
        "scheduler": {
            "_class_name": "RectifiedFlowScheduler",
            "_diffusers_version": "0.25.1",
            "num_train_timesteps": 1000,
            "sampler": "LinearQuadratic",
        },
        "vocoder": {
            "vocoder": {
                "upsample_initial_channel": 1536,
                "resblock": "AMP1",
                "upsample_rates": [5, 2, 2, 2, 2, 2],
                "resblock_kernel_sizes": [3, 7, 11],
                "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "stereo": True,
                "use_tanh_at_final": False,
                "activation": "snakebeta",
                "use_bias_at_final": False,
            },
            "bwe": {
                "upsample_initial_channel": 512,
                "resblock": "AMP1",
                "upsample_rates": [6, 5, 2, 2, 2],
                "resblock_kernel_sizes": [3, 7, 11],
                "upsample_kernel_sizes": [12, 11, 4, 4, 4],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "stereo": True,
                "use_tanh_at_final": False,
                "activation": "snakebeta",
                "use_bias_at_final": False,
                "apply_final_activation": False,
                "input_sampling_rate": 16000,
                "output_sampling_rate": 48000,
                "hop_length": 80,
                "n_fft": 512,
                "win_size": 512,
                "num_mels": 64,
            },
        },
    }


def _quantize_ltx_predicate(path: str, module: nn.Module) -> bool:
    if not hasattr(module, "to_quantized"):
        return False
    patterns = (
        ".to_q",
        ".to_k",
        ".to_v",
        ".to_out",
        ".ff.proj_in",
        ".ff.proj_out",
        ".audio_ff.proj_in",
        ".audio_ff.proj_out",
    )
    return any(path.endswith(pattern) for pattern in patterns)


def _to_component_dict(
    weights: Dict[str, mx.array], prefix: str
) -> Dict[str, mx.array]:
    start = f"{prefix}."
    return {k[len(start) :]: v for k, v in weights.items() if k.startswith(start)}


def _save_split_components(
    output_path: Path,
    unified_weights: Dict[str, mx.array],
    version: str,
    source_repo: Optional[str] = None,
    variant: str = "distilled",
    quantized_transformer: Optional[Dict[str, mx.array]] = None,
    q_bits: Optional[int] = None,
    q_group_size: Optional[int] = None,
) -> None:
    component_names = []
    transformer_weights = quantized_transformer or _to_component_dict(
        unified_weights, "transformer"
    )
    mx.save_safetensors(
        str(output_path / "transformer.safetensors"), transformer_weights
    )
    component_names.append("transformer")

    connector_weights = _to_component_dict(unified_weights, "connector")
    text_proj_weights = _to_component_dict(unified_weights, "text_embedding_projection")
    if text_proj_weights:
        connector_weights.update(
            {f"text_embedding_projection.{k}": v for k, v in text_proj_weights.items()}
        )
    if connector_weights:
        mx.save_safetensors(
            str(output_path / "connector.safetensors"), connector_weights
        )
        component_names.append("connector")

    for component in (
        "vae_decoder",
        "vae_encoder",
        "audio_vae",
        "vocoder",
    ):
        component_weights = _to_component_dict(unified_weights, component)
        if component_weights:
            mx.save_safetensors(
                str(output_path / f"{component}.safetensors"), component_weights
            )
            component_names.append(component)

    # Compatibility alias for runtime's unified upsampler lookup.
    upsampler_weights = _to_component_dict(unified_weights, "upsampler")
    if upsampler_weights:
        mx.save_safetensors(
            str(output_path / "upsampler.safetensors"), upsampler_weights
        )
        component_names.append("upsampler")
    for component in (
        "spatial_upscaler_x2_v1_1",
        "spatial_upscaler_x1_5_v1_0",
        "temporal_upscaler_x2_v1_0",
    ):
        component_weights = _to_component_dict(unified_weights, component)
        if component_weights:
            mx.save_safetensors(
                str(output_path / f"{component}.safetensors"), component_weights
            )
            component_names.append(component)

    manifest = {
        "format": "split",
        "model_version": "2.3.0" if version == "2.3" else "2.0.0",
        "components": component_names,
    }
    if source_repo:
        manifest["source"] = source_repo
    manifest["variant"] = variant
    if q_bits is not None and q_group_size is not None:
        manifest["quantized"] = True
        manifest["quantization_bits"] = q_bits
        manifest["quantization_group_size"] = q_group_size
        with open(output_path / "quantize_config.json", "w") as f:
            json.dump(
                {"quantization": {"bits": q_bits, "group_size": q_group_size}},
                f,
                indent=2,
            )
    with open(output_path / "split_model.json", "w") as f:
        json.dump(manifest, f, indent=2)


def quantize_transformer_weights(
    transformer_weights: Dict[str, mx.array],
    config: Dict[str, Any],
    q_bits: int,
    q_group_size: int,
) -> Dict[str, mx.array]:
    model = create_model_from_config(config)
    model.load_weights(list(transformer_weights.items()), strict=False)
    nn.quantize(
        model,
        bits=q_bits,
        group_size=q_group_size,
        class_predicate=_quantize_ltx_predicate,
    )
    return dict(mx_utils.tree_flatten(model.parameters()))


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    dtype: Optional[str] = None,
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
    include_audio: bool = True,
) -> Path:
    """Convert HuggingFace LTX-2 model to MLX format.

    Creates a unified model file containing:
    - Transformer weights (video + audio paths)
    - Video VAE decoder weights
    - Video VAE encoder weights
    - Audio VAE decoder weights (if include_audio=True)
    - Vocoder weights (if include_audio=True)

    Args:
        hf_path: HuggingFace model path or repo ID
        mlx_path: Output path for MLX model
        dtype: Target dtype (float16, float32, bfloat16)
        quantize: Whether to quantize the model
        q_bits: Quantization bits
        q_group_size: Quantization group size
        include_audio: Whether to include audio VAE and vocoder weights

    Returns:
        Path to converted model
    """
    print(f"Loading model from {hf_path}...")
    repo_hint_23 = "2.3" in hf_path.lower()
    allow_patterns = (
        [
            "ltx-2.3-22b-distilled.safetensors",
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
            "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
            "*.json",
            "config.json",
        ]
        if repo_hint_23
        else None
    )
    model_path = get_model_path(hf_path, allow_patterns=allow_patterns)
    model_version = detect_ltx_version(model_path, hf_path)
    print(f"Detected LTX version: {model_version}")

    # Load all raw weights
    print("Loading weights...")
    raw_weights = load_safetensors(model_path)

    # Build unified weights dictionary with proper prefixes
    unified_weights = {}

    # 1. Sanitize and add transformer weights
    print("Processing transformer weights...")
    transformer_weights = sanitize_transformer_weights(raw_weights)
    for k, v in transformer_weights.items():
        unified_weights[f"transformer.{k}"] = v
    print(f"  Added {len(transformer_weights)} transformer weights")

    # 2. Sanitize and add VAE decoder weights
    print("Processing VAE decoder weights...")
    vae_decoder_weights = sanitize_vae_weights(raw_weights)
    for k, v in vae_decoder_weights.items():
        unified_weights[f"vae_decoder.{k}"] = v
    print(f"  Added {len(vae_decoder_weights)} VAE decoder weights")

    # 3. Sanitize and add VAE encoder weights
    print("Processing VAE encoder weights...")
    vae_encoder_weights = sanitize_vae_encoder_weights(raw_weights)
    for k, v in vae_encoder_weights.items():
        unified_weights[f"vae_encoder.{k}"] = v
    print(f"  Added {len(vae_encoder_weights)} VAE encoder weights")

    if include_audio:
        # 4. Sanitize and add audio VAE decoder weights
        print("Processing audio VAE decoder weights...")
        audio_vae_weights = sanitize_audio_vae_weights(raw_weights)
        for k, v in audio_vae_weights.items():
            unified_weights[f"audio_vae.{k}"] = v
        print(f"  Added {len(audio_vae_weights)} audio VAE weights")

        # 5. Sanitize and add vocoder weights
        print("Processing vocoder weights...")
        vocoder_weights = sanitize_vocoder_weights(raw_weights)
        for k, v in vocoder_weights.items():
            unified_weights[f"vocoder.{k}"] = v
        print(f"  Added {len(vocoder_weights)} vocoder weights")

    # 6. Add text encoder projection weights (aggregate_embed)
    print("Processing text encoder projection weights...")
    text_proj_count = 0
    for k, v in raw_weights.items():
        if k.startswith("text_embedding_projection."):
            new_key = k  # Keep original naming
            unified_weights[new_key] = v
            text_proj_count += 1
    print(f"  Added {text_proj_count} text projection weights")

    # 7. Add connector weights (video and audio embeddings connectors)
    print("Processing connector weights...")
    connector_count = 0
    for k, v in raw_weights.items():
        if "video_embeddings_connector" in k or "audio_embeddings_connector" in k:
            # Keep under model.diffusion_model prefix for compatibility with generate_av.py
            if k.startswith("model.diffusion_model."):
                new_key = k.replace("model.diffusion_model.", "connector.")
            else:
                new_key = f"connector.{k}"
            unified_weights[new_key] = v
            connector_count += 1
    print(f"  Added {connector_count} connector weights")

    # 8. Add upsampler weights (enables fully self-contained unified/split model)
    upscaler_specs = (
        [
            ("upsampler", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            ("spatial_upscaler_x2_v1_1", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            (
                "spatial_upscaler_x1_5_v1_0",
                "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
            ),
            (
                "temporal_upscaler_x2_v1_0",
                "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
            ),
        ]
        if model_version == "2.3"
        else [("upsampler", "ltx-2-spatial-upscaler-x2-1.0.safetensors")]
    )
    for prefix, filename in upscaler_specs:
        upsampler_path = model_path / filename
        if not upsampler_path.exists():
            continue
        print(f"Processing {prefix} weights from {filename}...")
        upsampler_weights = mx.load(str(upsampler_path))
        for k, v in upsampler_weights.items():
            unified_weights[f"{prefix}.{k}"] = v
        print(f"  Added {len(upsampler_weights)} {prefix} weights")

    print(f"Total unified weights: {len(unified_weights)}")

    # Convert dtype if specified
    if dtype is not None:
        dtype_map = {
            "float16": mx.float16,
            "float32": mx.float32,
            "bfloat16": mx.bfloat16,
        }
        target_dtype = dtype_map.get(dtype, mx.float16)
        print(f"Converting to {dtype}...")
        unified_weights = {
            k: (
                v.astype(target_dtype)
                if v.dtype in [mx.float32, mx.float16, mx.bfloat16]
                else v
            )
            for k, v in unified_weights.items()
        }

    # Create output directory
    output_path = Path(mlx_path)
    output_path.mkdir(parents=True, exist_ok=True)

    config = build_output_config(version=model_version, include_audio=include_audio)

    config_out_path = output_path / "config.json"
    with open(config_out_path, "w") as f:
        json.dump(config, f, indent=2)

    embedded_config = build_embedded_config(model_version)
    if embedded_config:
        with open(output_path / "embedded_config.json", "w") as f:
            json.dump(embedded_config, f, indent=2)

    # Save weights in requested format
    print(f"Saving weights to {output_path}...")
    if quantize:
        transformer_raw = _to_component_dict(unified_weights, "transformer")
        quantized_transformer = quantize_transformer_weights(
            transformer_raw, config, q_bits=q_bits, q_group_size=q_group_size
        )
        _save_split_components(
            output_path=output_path,
            unified_weights=unified_weights,
            version=model_version,
            source_repo=hf_path,
            quantized_transformer=quantized_transformer,
            q_bits=q_bits,
            q_group_size=q_group_size,
        )
    else:
        save_weights(output_path, unified_weights)

    print(f"Model converted successfully to {output_path}")
    print(f"  - Transformer: {len(transformer_weights)} weights")
    print(f"  - VAE decoder: {len(vae_decoder_weights)} weights")
    print(f"  - VAE encoder: {len(vae_encoder_weights)} weights")
    if include_audio:
        print(f"  - Audio VAE: {len(audio_vae_weights)} weights")
        print(f"  - Vocoder: {len(vocoder_weights)} weights")

    return output_path


def save_weights(path: Path, weights: Dict[str, mx.array]) -> None:
    """Save weights in safetensors format.

    Args:
        path: Output directory
        weights: Dictionary of weights
    """
    # Use MLX's native save which supports bfloat16 directly
    output_file = path / "model.safetensors"
    mx.save_safetensors(str(output_file), weights)


def load_model(
    path_or_hf_repo: str,
    lazy: bool = False,
) -> LTXModel:
    """Load LTX model from path or HuggingFace.

    Args:
        path_or_hf_repo: Path to model or HuggingFace repo ID
        lazy: Whether to use lazy loading

    Returns:
        Loaded LTXModel
    """
    model_path = get_model_path(path_or_hf_repo)

    # Load config
    config = load_config(model_path)

    # Create model
    model = create_model_from_config(config)

    # Load weights
    weights = load_safetensors(model_path)

    # Sanitize if needed
    weights = sanitize_weights(weights)

    # Load weights into model
    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    return model


def main():
    """CLI entry point for model conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert LTX-2/LTX-2.3 model to MLX format with audio support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with audio support (default)
  python -m mlx_video.convert --hf-path Lightricks/LTX-2.3 --mlx-path mlx_model

  # Convert video-only (no audio)
  python -m mlx_video.convert --hf-path Lightricks/LTX-2.3 --mlx-path mlx_model --no-audio

  # Convert quantized Q4 split model
  python -m mlx_video.convert --hf-path Lightricks/LTX-2.3 --mlx-path mlx_model_q4 --quantize --q-bits 4
        """,
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        default="Lightricks/LTX-2.3",
        help="HuggingFace model path or repo ID (default: Lightricks/LTX-2.3)",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="Output path for MLX model (default: mlx_model)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "bfloat16"],
        default="bfloat16",
        help="Target dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Exclude audio VAE and vocoder weights (video-only model)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the model (experimental)",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        help="Quantization bits (default: 4)",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )

    args = parser.parse_args()

    convert(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        dtype=args.dtype,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        include_audio=not args.no_audio,
    )


if __name__ == "__main__":
    main()
