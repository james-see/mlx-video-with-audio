"""Audio-Video generation pipeline for LTX-2."""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mx_utils
import numpy as np
from tqdm import tqdm


# ANSI color codes
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType
from mlx_video.models.ltx.ltx import LTXModel
from mlx_video.models.ltx.transformer import Modality
from mlx_video.convert import (
    sanitize_transformer_weights,
    sanitize_audio_vae_weights,
    sanitize_vocoder_weights,
)
from mlx_video.utils import (
    to_denoised,
    get_model_path,
    load_image,
    prepare_image_for_encoding,
)
from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder
from mlx_video.models.ltx.video_vae.encoder import load_vae_encoder
from mlx_video.models.ltx.video_vae.tiling import TilingConfig
from mlx_video.models.ltx.upsampler import load_upsampler, upsample_latents
from mlx_video.conditioning import VideoConditionByLatentIndex, apply_conditioning
from mlx_video.conditioning.latent import LatentState, apply_denoise_mask


# Distilled sigma schedules
DEFAULT_STAGE_1_SIGMAS = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]
DEFAULT_STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]
BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3d cgi look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or ai artifacts."
)


def linear_quadratic_schedule(
    num_steps: int, threshold_noise: float = 0.025, linear_steps: Optional[int] = None
) -> list[float]:
    """Match the official Lightricks LinearQuadratic sigma schedule."""
    if num_steps <= 1:
        return [1.0]
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [
        i * threshold_noise / linear_steps for i in range(linear_steps)
    ]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (
        quadratic_steps**2
    )
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule[:-1]


def ltx2_schedule(
    num_steps: int,
    tokens: Optional[int],
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> list[float]:
    """Match official LTX2Scheduler with token-dependent sigma shifting."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")

    token_count = tokens if tokens is not None else MAX_SHIFT_ANCHOR
    sigmas = np.linspace(1.0, 0.0, num_steps + 1, dtype=np.float64)

    x1 = BASE_SHIFT_ANCHOR
    x2 = MAX_SHIFT_ANCHOR
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    sigma_shift = token_count * mm + b

    non_zero_mask = sigmas != 0
    exp_shift = np.exp(sigma_shift)
    sigmas[non_zero_mask] = exp_shift / (
        exp_shift + (1.0 / sigmas[non_zero_mask] - 1.0)
    )

    if stretch:
        non_zero_sigmas = sigmas[non_zero_mask]
        if non_zero_sigmas.size > 0:
            one_minus = 1.0 - non_zero_sigmas
            scale_factor = one_minus[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus / scale_factor)
            sigmas[non_zero_mask] = stretched

    return [float(x) for x in sigmas.astype(np.float32)]


def build_stage_sigma_schedules(
    num_inference_steps: int,
    stage1_token_count: Optional[int] = None,
    use_ltx2_scheduler: bool = True,
) -> tuple[list[float], list[float]]:
    """Derive stage schedules from one total step count, matching app semantics."""
    if num_inference_steps < 2:
        raise ValueError("num_inference_steps must be at least 2")
    if num_inference_steps == 11:
        return DEFAULT_STAGE_1_SIGMAS, DEFAULT_STAGE_2_SIGMAS

    stage1_steps = max(2, round(num_inference_steps * 8 / 11))
    stage2_steps = max(1, num_inference_steps - stage1_steps)
    if stage1_steps + stage2_steps != num_inference_steps:
        stage2_steps = num_inference_steps - stage1_steps

    if use_ltx2_scheduler and stage1_token_count is not None:
        full_schedule = ltx2_schedule(num_inference_steps, tokens=stage1_token_count)
    else:
        full_schedule = linear_quadratic_schedule(num_inference_steps)
    stage1_sigmas = full_schedule[: stage1_steps + 1]
    stage2_sigmas = full_schedule[stage1_steps:]
    return stage1_sigmas, stage2_sigmas


# Default HuggingFace model for text encoder (only used when NOT using unified MLX model)
DEFAULT_HF_MODEL = "Lightricks/LTX-2"
# Default MLX Gemma for text encoder when using unified model (quality-first)
DEFAULT_UNIFIED_TEXT_ENCODER = "mlx-community/gemma-3-12b-it-bf16"


def is_unified_mlx_model(model_path: Path) -> bool:
    """Check if model_path contains a unified MLX model.

    Unified MLX models come in two layouts:
    1. Single-file: model.safetensors + config.json
    2. Split-weight: transformer.safetensors + vae_decoder.safetensors + config.json
       with model_type "AudioVideo" (e.g. quantized/distilled repos)

    Neither layout contains ltx-2-19b-distilled.safetensors (HuggingFace format).
    """
    model_path = Path(model_path)
    has_config = (model_path / "config.json").exists()
    has_hf_format = (model_path / "ltx-2-19b-distilled.safetensors").exists()
    if not has_config or has_hf_format:
        return False
    has_single = (model_path / "model.safetensors").exists()
    if has_single:
        return True
    split_manifest = model_path / "split_model.json"
    if split_manifest.exists():
        try:
            with open(split_manifest, "r") as f:
                manifest = json.load(f)
            return manifest.get("format") == "split"
        except (json.JSONDecodeError, OSError):
            pass
    has_split = (model_path / "transformer.safetensors").exists() and (
        model_path / "vae_decoder.safetensors"
    ).exists()
    if has_split:
        try:
            with open(model_path / "config.json", "r") as f:
                cfg = json.load(f)
            return cfg.get("model_type") == "AudioVideo"
        except (json.JSONDecodeError, OSError):
            return False
    return False


def _looks_like_text_config(config_dict: dict) -> bool:
    required = {
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    }
    if "text_config" in config_dict and isinstance(config_dict["text_config"], dict):
        return True
    return required.issubset(set(config_dict.keys()))


def _is_ltx_23_model(model_repo: Optional[str], model_path: Path) -> bool:
    """Best-effort check for LTX-2.3 checkpoints to choose compatible upsamplers."""
    if model_repo and "2.3" in model_repo.lower():
        return True
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        return False
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        model_version = str(cfg.get("model_version", ""))
        return model_version.startswith("2.3")
    except (json.JSONDecodeError, OSError):
        return False


def validate_text_encoder_config(text_encoder_path: Path) -> None:
    """Validate that the resolved text encoder path contains a usable config."""
    config_file = text_encoder_path / "config.json"
    if not config_file.exists():
        raise ValueError(f"Text encoder config not found at {config_file}")

    with open(config_file, "r") as f:
        config_dict = json.load(f)

    keys = sorted(config_dict.keys())
    print(f"TEXT_ENCODER:PATH:{text_encoder_path}", file=sys.stderr, flush=True)
    print(
        f"TEXT_ENCODER:CONFIG_KEYS:{','.join(keys[:20])}",
        file=sys.stderr,
        flush=True,
    )
    if not _looks_like_text_config(config_dict):
        key_preview = ", ".join(keys[:20]) if keys else "<none>"
        message = (
            "Text encoder config is missing `text_config`. "
            f"Resolved path: {text_encoder_path}. "
            f"Top-level keys: [{key_preview}]. "
            "Use --text-encoder-repo mlx-community/gemma-3-12b-it-bf16 or update mlx-video-with-audio."
        )
        print(f"TEXT_ENCODER_CONFIG_ERROR:{message}", file=sys.stderr, flush=True)
        raise ValueError(message)


def load_unified_weights(model_path: Path, prefix: str) -> dict:
    """Load weights from unified MLX model with given prefix.

    Supports two layouts:
    1. Single-file (model.safetensors): filter by prefix and strip it.
    2. Split-weight (transformer.safetensors, vae_decoder.safetensors, etc.):
       load the matching file directly — keys have no prefix.

    Args:
        model_path: Path to unified model directory
        prefix: Prefix to filter weights (e.g., 'transformer.', 'vae_decoder.')

    Returns:
        Dictionary of weights with prefix stripped
    """
    single_file = model_path / "model.safetensors"
    if single_file.exists():
        all_weights = mx.load(str(single_file))
        return {
            k[len(prefix) :]: v for k, v in all_weights.items() if k.startswith(prefix)
        }
    split_name = prefix.rstrip(".")
    split_file = model_path / f"{split_name}.safetensors"
    if split_file.exists():
        raw = mx.load(str(split_file))
        stripped = {}
        for k, v in raw.items():
            stripped[k[len(prefix) :] if k.startswith(prefix) else k] = v
        return stripped
    return {}


# Audio constants
AUDIO_SAMPLE_RATE = 24000  # Output audio sample rate
AUDIO_LATENT_SAMPLE_RATE = 16000  # VAE internal sample rate
AUDIO_HOP_LENGTH = 160
AUDIO_LATENT_DOWNSAMPLE_FACTOR = 4
AUDIO_LATENT_CHANNELS = 8  # Latent channels before patchifying
AUDIO_MEL_BINS = 16
AUDIO_LATENTS_PER_SECOND = (
    AUDIO_LATENT_SAMPLE_RATE / AUDIO_HOP_LENGTH / AUDIO_LATENT_DOWNSAMPLE_FACTOR
)  # 25


def create_video_position_grid(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    temporal_scale: int = 8,
    spatial_scale: int = 32,
    fps: float = 24.0,
    causal_fix: bool = True,
) -> mx.array:
    """Create position grid for video RoPE in pixel space."""
    patch_size_t, patch_size_h, patch_size_w = 1, 1, 1

    t_coords = np.arange(0, num_frames, patch_size_t)
    h_coords = np.arange(0, height, patch_size_h)
    w_coords = np.arange(0, width, patch_size_w)

    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
    patch_starts = np.stack([t_grid, h_grid, w_grid], axis=0)

    patch_size_delta = np.array([patch_size_t, patch_size_h, patch_size_w]).reshape(
        3, 1, 1, 1
    )
    patch_ends = patch_starts + patch_size_delta

    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)
    num_patches = num_frames * height * width
    latent_coords = latent_coords.reshape(3, num_patches, 2)
    latent_coords = np.tile(latent_coords[np.newaxis, ...], (batch_size, 1, 1, 1))

    scale_factors = np.array([temporal_scale, spatial_scale, spatial_scale]).reshape(
        1, 3, 1, 1
    )
    pixel_coords = (latent_coords * scale_factors).astype(np.float32)

    if causal_fix:
        pixel_coords[:, 0, :, :] = np.clip(
            pixel_coords[:, 0, :, :] + 1 - temporal_scale, a_min=0, a_max=None
        )

    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps

    return mx.array(pixel_coords, dtype=mx.float32)


def create_audio_position_grid(
    batch_size: int,
    audio_frames: int,
    sample_rate: int = AUDIO_LATENT_SAMPLE_RATE,
    hop_length: int = AUDIO_HOP_LENGTH,
    downsample_factor: int = AUDIO_LATENT_DOWNSAMPLE_FACTOR,
    is_causal: bool = True,
) -> mx.array:
    """Create temporal position grid for audio RoPE.

    Audio positions are timestamps in seconds, shape (B, 1, T, 2).
    Matches PyTorch's AudioPatchifier.get_patch_grid_bounds exactly.
    """

    def get_audio_latent_time_in_sec(start_idx: int, end_idx: int) -> np.ndarray:
        """Convert latent indices to seconds (matching PyTorch's _get_audio_latent_time_in_sec)."""
        latent_frame = np.arange(start_idx, end_idx, dtype=np.float32)
        mel_frame = latent_frame * downsample_factor
        if is_causal:
            # Frame offset for causal alignment (PyTorch uses +1 - downsample_factor)
            mel_frame = np.clip(mel_frame + 1 - downsample_factor, 0, None)
        return mel_frame * hop_length / sample_rate

    # Start times: latent indices 0 to audio_frames
    start_times = get_audio_latent_time_in_sec(0, audio_frames)

    # End times: latent indices 1 to audio_frames+1 (shifted by 1)
    end_times = get_audio_latent_time_in_sec(1, audio_frames + 1)

    # Shape: (B, 1, T, 2)
    positions = np.stack([start_times, end_times], axis=-1)
    positions = positions[np.newaxis, np.newaxis, :, :]  # (1, 1, T, 2)
    positions = np.tile(positions, (batch_size, 1, 1, 1))

    return mx.array(positions, dtype=mx.float32)


def compute_audio_frames(num_video_frames: int, fps: float) -> int:
    """Compute number of audio latent frames given video duration."""
    duration = num_video_frames / fps
    return round(duration * AUDIO_LATENTS_PER_SECOND)


def _to_velocity(
    noisy: mx.array,
    denoised: mx.array,
    sigma: float,
    calc_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    sigma_arr = mx.array(sigma, dtype=calc_dtype)
    return (
        (noisy.astype(calc_dtype) - denoised.astype(calc_dtype)) / sigma_arr
    ).astype(noisy.dtype)


def _euler_step(
    sample: mx.array,
    denoised_sample: mx.array,
    sigma: float,
    sigma_next: float,
    calc_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    velocity = _to_velocity(sample, denoised_sample, sigma, calc_dtype=calc_dtype)
    dt = mx.array(sigma_next - sigma, dtype=calc_dtype)
    stepped = sample.astype(calc_dtype) + velocity.astype(calc_dtype) * dt
    return stepped.astype(sample.dtype)


def denoise_av(
    video_latents: mx.array,
    audio_latents: mx.array,
    video_positions: mx.array,
    audio_positions: mx.array,
    video_embeddings: mx.array,
    audio_embeddings: mx.array,
    video_embeddings_negative: Optional[mx.array],
    audio_embeddings_negative: Optional[mx.array],
    transformer: LTXModel,
    sigmas: list,
    verbose: bool = True,
    video_state: Optional[LatentState] = None,
    stage: int = 1,
    audio_enabled: bool = True,
    cfg_scale: float = 1.0,
    use_gradient_estimation: bool = False,
    ge_gamma: float = 2.0,
) -> tuple[mx.array, mx.array]:
    """Run denoising loop for audio-video generation with optional I2V conditioning.

    Args:
        video_latents: Video latent tensor (B, C, F, H, W)
        audio_latents: Audio latent tensor (B, C, T, F)
        video_positions: Video position embeddings
        audio_positions: Audio position embeddings
        video_embeddings: Video text embeddings
        audio_embeddings: Audio text embeddings
        video_embeddings_negative: Negative video text embeddings for CFG
        audio_embeddings_negative: Negative audio text embeddings for CFG
        transformer: LTX model
        sigmas: List of sigma values
        verbose: Whether to show progress bar
        video_state: Optional LatentState for I2V conditioning
        stage: Stage number for progress logging (1 or 2)
        audio_enabled: Whether to enable the audio branch and AV coupling
        cfg_scale: Classifier-free guidance scale (1.0 disables CFG)
        use_gradient_estimation: Whether to use gradient estimating Euler updates
        ge_gamma: Gradient estimation coefficient

    Returns:
        Tuple of (video_latents, audio_latents)
    """
    dtype = video_latents.dtype
    # If video state is provided, use its latent
    if video_state is not None:
        video_latents = video_state.latent

    cfg_enabled = (
        cfg_scale > 1.0
        and video_embeddings_negative is not None
        and audio_embeddings_negative is not None
    )
    previous_video_velocity = None
    previous_audio_velocity = None

    total_steps = len(sigmas) - 1
    for i in tqdm(range(total_steps), desc="Denoising A/V", disable=not verbose):
        # Emit structured progress for external parsers
        print(
            f"STAGE:{stage}:STEP:{i + 1}:{total_steps}:Denoising",
            file=sys.stderr,
            flush=True,
        )
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        # Flatten video latents
        b, c, f, h, w = video_latents.shape
        num_video_tokens = f * h * w
        video_flat = mx.transpose(mx.reshape(video_latents, (b, c, -1)), (0, 2, 1))

        # Flatten audio latents: (B, C, T, F) -> (B, T, C*F)
        ab, ac, at, af = audio_latents.shape
        audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))  # (B, T, C, F)
        audio_flat = mx.reshape(audio_flat, (ab, at, ac * af))

        # Compute per-token timesteps for video
        # For I2V: conditioned tokens get timestep=0 (mask=0), unconditioned get timestep=sigma (mask=1)
        if video_state is not None:
            # Reshape denoise_mask from (B, 1, F, 1, 1) to (B, num_tokens)
            denoise_mask_flat = mx.reshape(video_state.denoise_mask, (b, 1, f, 1, 1))
            denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
            denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_video_tokens))
            # Per-token timesteps: sigma * mask
            video_timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
        else:
            # All tokens get the same timestep
            video_timesteps = mx.full((b, num_video_tokens), sigma, dtype=dtype)

        video_modality = Modality(
            latent=video_flat,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_embeddings,
            sigma=None,
            context_mask=None,
            enabled=True,
        )

        audio_modality = (
            Modality(
                latent=audio_flat,
                timesteps=mx.full((ab, at), sigma, dtype=dtype),
                positions=audio_positions,
                context=audio_embeddings,
                sigma=None,
                context_mask=None,
                enabled=True,
            )
            if audio_enabled
            else None
        )
        video_velocity_pos, audio_velocity_pos = transformer(
            video=video_modality, audio=audio_modality
        )
        if audio_velocity_pos is not None:
            mx.eval(video_velocity_pos, audio_velocity_pos)
        else:
            mx.eval(video_velocity_pos)

        video_velocity_neg = None
        audio_velocity_neg = None
        if cfg_enabled:
            video_modality_negative = Modality(
                latent=video_flat,
                timesteps=video_timesteps,
                positions=video_positions,
                context=video_embeddings_negative,
                sigma=None,
                context_mask=None,
                enabled=True,
            )
            audio_modality_negative = (
                Modality(
                    latent=audio_flat,
                    timesteps=mx.full((ab, at), sigma, dtype=dtype),
                    positions=audio_positions,
                    context=audio_embeddings_negative,
                    sigma=None,
                    context_mask=None,
                    enabled=True,
                )
                if audio_enabled
                else None
            )
            video_velocity_neg, audio_velocity_neg = transformer(
                video=video_modality_negative, audio=audio_modality_negative
            )
            if audio_velocity_neg is not None:
                mx.eval(video_velocity_neg, audio_velocity_neg)
            else:
                mx.eval(video_velocity_neg)

        # Reshape velocities back
        video_velocity_pos = mx.reshape(
            mx.transpose(video_velocity_pos, (0, 2, 1)), (b, c, f, h, w)
        )
        if audio_velocity_pos is not None:
            audio_velocity_pos = mx.reshape(audio_velocity_pos, (ab, at, ac, af))
            audio_velocity_pos = mx.transpose(audio_velocity_pos, (0, 2, 1, 3))
        if video_velocity_neg is not None:
            video_velocity_neg = mx.reshape(
                mx.transpose(video_velocity_neg, (0, 2, 1)), (b, c, f, h, w)
            )
        if audio_velocity_neg is not None:
            audio_velocity_neg = mx.reshape(audio_velocity_neg, (ab, at, ac, af))
            audio_velocity_neg = mx.transpose(audio_velocity_neg, (0, 2, 1, 3))

        # Compute denoised
        video_denoised = to_denoised(video_latents, video_velocity_pos, sigma)
        audio_denoised = (
            to_denoised(audio_latents, audio_velocity_pos, sigma)
            if audio_velocity_pos is not None
            else audio_latents
        )
        if cfg_enabled and video_velocity_neg is not None:
            video_denoised_negative = to_denoised(
                video_latents, video_velocity_neg, sigma
            )
            cfg_delta = mx.array(cfg_scale - 1.0, dtype=video_denoised.dtype)
            video_denoised = video_denoised + cfg_delta * (
                video_denoised - video_denoised_negative
            )
            if audio_velocity_neg is not None:
                audio_denoised_negative = to_denoised(
                    audio_latents, audio_velocity_neg, sigma
                )
                audio_cfg_delta = mx.array(cfg_scale - 1.0, dtype=audio_denoised.dtype)
                audio_denoised = audio_denoised + audio_cfg_delta * (
                    audio_denoised - audio_denoised_negative
                )

        # Apply conditioning mask for video if state is provided
        if video_state is not None:
            video_denoised = apply_denoise_mask(
                video_denoised, video_state.clean_latent, video_state.denoise_mask
            )

        mx.eval(video_denoised, audio_denoised)

        # Optional gradient-estimating correction before the Euler step.
        if use_gradient_estimation and sigma_next > 0:
            current_video_velocity = _to_velocity(video_latents, video_denoised, sigma)
            if previous_video_velocity is not None:
                delta_velocity = current_video_velocity - previous_video_velocity
                total_velocity = (
                    mx.array(ge_gamma, dtype=current_video_velocity.dtype)
                    * delta_velocity
                    + previous_video_velocity
                )
                video_denoised = to_denoised(video_latents, total_velocity, sigma)
            previous_video_velocity = current_video_velocity

            if audio_velocity_pos is not None:
                current_audio_velocity = _to_velocity(
                    audio_latents, audio_denoised, sigma
                )
                if previous_audio_velocity is not None:
                    delta_audio_velocity = (
                        current_audio_velocity - previous_audio_velocity
                    )
                    total_audio_velocity = (
                        mx.array(ge_gamma, dtype=current_audio_velocity.dtype)
                        * delta_audio_velocity
                        + previous_audio_velocity
                    )
                    audio_denoised = to_denoised(
                        audio_latents, total_audio_velocity, sigma
                    )
                previous_audio_velocity = current_audio_velocity

        # Official Euler step in float32 for stability.
        if sigma_next > 0:
            video_latents = _euler_step(
                video_latents, video_denoised, sigma, sigma_next
            )
            if audio_velocity_pos is not None:
                audio_latents = _euler_step(
                    audio_latents, audio_denoised, sigma, sigma_next
                )
        else:
            video_latents = video_denoised
            if audio_velocity_pos is not None:
                audio_latents = audio_denoised
        mx.eval(video_latents, audio_latents)

    return video_latents, audio_latents


def load_audio_decoder(model_path: Path, use_unified: bool = False):
    """Load audio VAE decoder.

    Args:
        model_path: Path to model directory
        use_unified: If True, load from unified MLX format
    """
    from mlx_video.models.ltx.audio_vae import AudioDecoder, CausalityAxis, NormType

    embedded_audio_cfg = {}
    embedded_stft_cfg = {}
    embedded_cfg_path = model_path / "embedded_config.json"
    if embedded_cfg_path.exists():
        try:
            with open(embedded_cfg_path, "r") as f:
                embedded_root_cfg = json.load(f)
            embedded_audio_cfg = embedded_root_cfg.get("audio_vae", {}).get(
                "preprocessing", {}
            )
            embedded_stft_cfg = embedded_audio_cfg.get("stft", {})
        except (json.JSONDecodeError, OSError):
            pass

    decoder = AudioDecoder(
        ch=128,
        out_ch=2,  # stereo
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions={8, 16, 32},
        resolution=256,
        z_channels=AUDIO_LATENT_CHANNELS,
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT,
        mel_bins=64,  # Output mel bins
    )
    if use_unified:
        # Load from unified MLX model (weights already sanitized)
        sanitized = load_unified_weights(model_path, "audio_vae.")
        if sanitized:
            decoder.load_weights(list(sanitized.items()), strict=False)
            # Manually load per-channel statistics
            if "per_channel_statistics._mean_of_means" in sanitized:
                decoder.per_channel_statistics._mean_of_means = sanitized[
                    "per_channel_statistics._mean_of_means"
                ]
            if "per_channel_statistics._std_of_means" in sanitized:
                decoder.per_channel_statistics._std_of_means = sanitized[
                    "per_channel_statistics._std_of_means"
                ]
    else:
        # Load from HuggingFace format (needs sanitization)
        weight_file = model_path / "ltx-2-19b-distilled.safetensors"
        if weight_file.exists():
            raw_weights = mx.load(str(weight_file))
            sanitized = sanitize_audio_vae_weights(raw_weights)
            if sanitized:
                decoder.load_weights(list(sanitized.items()), strict=False)
                # Manually load per-channel statistics
                if "per_channel_statistics._mean_of_means" in sanitized:
                    decoder.per_channel_statistics._mean_of_means = sanitized[
                        "per_channel_statistics._mean_of_means"
                    ]
                if "per_channel_statistics._std_of_means" in sanitized:
                    decoder.per_channel_statistics._std_of_means = sanitized[
                        "per_channel_statistics._std_of_means"
                    ]

    return decoder


def load_vocoder(model_path: Path, use_unified: bool = False):
    """Load vocoder for mel to waveform conversion.

    Args:
        model_path: Path to model directory
        use_unified: If True, load from unified MLX format
    """
    from mlx_video.models.ltx.audio_vae import BigVGANVocoder, Vocoder, VocoderWithBWE

    resblock_kernel_sizes = [3, 7, 11]
    upsample_rates = [6, 5, 2, 2, 2]
    upsample_kernel_sizes = [16, 15, 8, 4, 4]
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    upsample_initial_channel = 1024
    vocoder_resblock = "1"
    vocoder_activation = "leaky_relu"
    use_tanh_at_final = True
    use_bias_at_final = True
    apply_final_activation = True
    bwe_cfg = None
    embedded_cfg_path = model_path / "embedded_config.json"
    if embedded_cfg_path.exists():
        try:
            with open(embedded_cfg_path, "r") as f:
                embedded_cfg = json.load(f)
            voc_cfg = embedded_cfg.get("vocoder", {}).get("vocoder", {})
            if voc_cfg:
                upsample_initial_channel = voc_cfg.get(
                    "upsample_initial_channel", upsample_initial_channel
                )
                upsample_rates = voc_cfg.get("upsample_rates", upsample_rates)
                upsample_kernel_sizes = voc_cfg.get(
                    "upsample_kernel_sizes", upsample_kernel_sizes
                )
                resblock_kernel_sizes = voc_cfg.get(
                    "resblock_kernel_sizes", resblock_kernel_sizes
                )
                resblock_dilation_sizes = voc_cfg.get(
                    "resblock_dilation_sizes", resblock_dilation_sizes
                )
                vocoder_resblock = str(voc_cfg.get("resblock", vocoder_resblock))
                vocoder_activation = str(
                    voc_cfg.get("activation", vocoder_activation)
                ).lower()
                use_tanh_at_final = bool(
                    voc_cfg.get("use_tanh_at_final", use_tanh_at_final)
                )
                use_bias_at_final = bool(
                    voc_cfg.get("use_bias_at_final", use_bias_at_final)
                )
                apply_final_activation = bool(
                    voc_cfg.get("apply_final_activation", apply_final_activation)
                )
            bwe_cfg = embedded_cfg.get("vocoder", {}).get("bwe", None)
        except (json.JSONDecodeError, OSError):
            pass

    is_bigvgan = vocoder_activation == "snakebeta" or vocoder_resblock.upper() == "AMP1"
    if is_bigvgan:
        core_vocoder = BigVGANVocoder(
            resblock_kernel_sizes=resblock_kernel_sizes,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_initial_channel=upsample_initial_channel,
            stereo=True,
            output_sample_rate=AUDIO_SAMPLE_RATE,
            use_tanh_at_final=use_tanh_at_final,
            use_bias_at_final=use_bias_at_final,
            apply_final_activation=apply_final_activation,
        )
        if bwe_cfg:
            bwe_vocoder = BigVGANVocoder(
                resblock_kernel_sizes=bwe_cfg.get(
                    "resblock_kernel_sizes", resblock_kernel_sizes
                ),
                upsample_rates=bwe_cfg.get("upsample_rates", upsample_rates),
                upsample_kernel_sizes=bwe_cfg.get(
                    "upsample_kernel_sizes", upsample_kernel_sizes
                ),
                resblock_dilation_sizes=bwe_cfg.get(
                    "resblock_dilation_sizes", resblock_dilation_sizes
                ),
                upsample_initial_channel=bwe_cfg.get(
                    "upsample_initial_channel", upsample_initial_channel
                ),
                stereo=bool(bwe_cfg.get("stereo", True)),
                output_sample_rate=int(
                    bwe_cfg.get("output_sampling_rate", AUDIO_SAMPLE_RATE)
                ),
                use_tanh_at_final=bool(bwe_cfg.get("use_tanh_at_final", False)),
                use_bias_at_final=bool(bwe_cfg.get("use_bias_at_final", False)),
                apply_final_activation=bool(
                    bwe_cfg.get("apply_final_activation", False)
                ),
            )
            vocoder = VocoderWithBWE(
                vocoder=core_vocoder,
                bwe_generator=bwe_vocoder,
                input_sampling_rate=int(
                    bwe_cfg.get("input_sampling_rate", AUDIO_SAMPLE_RATE)
                ),
                output_sampling_rate=int(
                    bwe_cfg.get("output_sampling_rate", AUDIO_SAMPLE_RATE)
                ),
                hop_length=int(bwe_cfg.get("hop_length", 80)),
                win_length=int(bwe_cfg.get("win_size", 512)),
            )
        else:
            vocoder = core_vocoder
    else:
        vocoder = Vocoder(
            resblock_kernel_sizes=resblock_kernel_sizes,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_initial_channel=upsample_initial_channel,
            stereo=True,
            output_sample_rate=AUDIO_SAMPLE_RATE,
        )
    if use_unified:
        # Load from unified MLX model (weights already sanitized)
        sanitized = load_unified_weights(model_path, "vocoder.")
        if sanitized:
            # Distilled split vocoder uses PyTorch layout for ConvTranspose1d upsamplers
            # while other conv weights are already MLX-compatible.
            fixed = {}
            for key, value in sanitized.items():
                if value.ndim == 3 and (
                    key.startswith("ups.") or key.startswith("bwe_generator.ups.")
                ):
                    value = mx.transpose(value, (1, 2, 0))
                fixed[key] = value
            sanitized = fixed
            if isinstance(vocoder, VocoderWithBWE):
                mapped = {}
                for key, value in sanitized.items():
                    if key.startswith("bwe_generator.") or key.startswith("mel_stft."):
                        mapped[key] = value
                    else:
                        mapped[f"vocoder.{key}"] = value
                sanitized = mapped
            vocoder.load_weights(list(sanitized.items()), strict=False)
    else:
        # Load from HuggingFace format (needs sanitization)
        weight_file = model_path / "ltx-2-19b-distilled.safetensors"
        if weight_file.exists():
            raw_weights = mx.load(str(weight_file))
            sanitized = sanitize_vocoder_weights(raw_weights)
            if sanitized:
                vocoder.load_weights(list(sanitized.items()), strict=False)

    return vocoder


def save_audio(audio: np.ndarray, path: Path, sample_rate: int = AUDIO_SAMPLE_RATE):
    """Save audio to WAV file."""
    import wave

    # Ensure audio is in correct format (channels, samples) or (samples,)
    if audio.ndim == 2:
        # (channels, samples) -> (samples, channels)
        audio = audio.T

    # Peak-normalize so the loudest sample hits ±0.95, then convert to int16
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio * (0.95 / peak)
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2 if audio_int16.ndim == 2 else 1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path):
    """Combine video and audio into final output using ffmpeg."""
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}FFmpeg error: {e.stderr.decode()}{Colors.RESET}")
        return False
    except FileNotFoundError:
        print(f"{Colors.RED}FFmpeg not found. Please install ffmpeg.{Colors.RESET}")
        return False


def generate_video_with_audio(
    model_repo: str,
    text_encoder_repo: Optional[str],
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_frames: int = 33,
    seed: int = 42,
    fps: int = 24,
    output_path: str = "output_av.mp4",
    output_audio_path: Optional[str] = None,
    save_audio_separately: bool = False,
    negative_prompt: Optional[str] = DEFAULT_NEGATIVE_PROMPT,
    cfg_scale: float = 3.0,
    verbose: bool = True,
    enhance_prompt: bool = False,
    use_uncensored_enhancer: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.7,
    image: Optional[str] = None,
    image_strength: float = 1.0,
    image_frame_idx: int = 0,
    tiling: str = "auto",
    num_inference_steps: int = 30,
):
    """Generate video with synchronized audio from text prompt, optionally conditioned on an image.

    Args:
        model_repo: Model repository ID
        text_encoder_repo: Text encoder repository ID
        prompt: Text description of the video to generate
        height: Output video height (must be divisible by 64)
        width: Output video width (must be divisible by 64)
        num_frames: Number of frames
        seed: Random seed
        fps: Frames per second
        output_path: Output video path
        output_audio_path: Output audio path
        negative_prompt: Optional negative prompt for CFG
        cfg_scale: CFG scale (1.0 disables CFG)
        verbose: Whether to print progress
        enhance_prompt: Whether to enhance prompt using Gemma
        max_tokens: Max tokens for prompt enhancement
        temperature: Temperature for prompt enhancement
        image: Path to conditioning image for I2V
        image_strength: Conditioning strength (1.0 = full denoise)
        image_frame_idx: Frame index to condition (0 = first frame)
        tiling: Tiling mode for VAE decoding (auto/none/default/aggressive/conservative/spatial/temporal)
        num_inference_steps: Total denoising steps across both stages
    """
    start_time = time.time()

    # Validate dimensions
    assert height % 64 == 0, f"Height must be divisible by 64, got {height}"
    assert width % 64 == 0, f"Width must be divisible by 64, got {width}"

    if num_frames % 8 != 1:
        adjusted_num_frames = round((num_frames - 1) / 8) * 8 + 1
        print(
            f"{Colors.YELLOW}⚠️  Adjusted frames to {adjusted_num_frames}{Colors.RESET}"
        )
        num_frames = adjusted_num_frames

    # Calculate audio frames
    audio_frames = compute_audio_frames(num_frames, fps)

    is_i2v = image is not None
    mode_str = "I2V+Audio" if is_i2v else "T2V+Audio"
    print(
        f"{Colors.BOLD}{Colors.CYAN}🎬 [{mode_str}] Generating {width}x{height} video with {num_frames} frames + audio{Colors.RESET}"
    )
    print(
        f"{Colors.DIM}Audio: {audio_frames} latent frames @ {AUDIO_SAMPLE_RATE}Hz{Colors.RESET}"
    )
    print(
        f"{Colors.DIM}Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}{Colors.RESET}"
    )
    if is_i2v:
        print(
            f"{Colors.DIM}Image: {image} (strength={image_strength}, frame={image_frame_idx}){Colors.RESET}"
        )

    model_path = get_model_path(model_repo)

    # Check if using unified MLX model format
    use_unified = is_unified_mlx_model(model_path)
    if use_unified:
        print(
            f"{Colors.DIM}Using unified MLX model format (no Lightricks download){Colors.RESET}"
        )
        # Use unified model path for everything - VAE, upsampler, connectors from model.safetensors
        # Use official bf16 Gemma text encoder for quality/compatibility
        hf_model_path = model_path
        text_encoder_path = get_model_path(
            text_encoder_repo or DEFAULT_UNIFIED_TEXT_ENCODER
        )
        vae_model_path = model_path
    else:
        text_encoder_path = (
            model_path
            if text_encoder_repo is None
            else get_model_path(text_encoder_repo)
        )
        hf_model_path = model_path  # For upsampler, VAE, etc.
        vae_model_path = model_path

    print(f"RESOLVE:MODEL_PATH:{model_path}", file=sys.stderr, flush=True)
    print(f"RESOLVE:USE_UNIFIED:{use_unified}", file=sys.stderr, flush=True)
    print(f"RESOLVE:TEXT_ENCODER_PATH:{text_encoder_path}", file=sys.stderr, flush=True)
    validate_text_encoder_config(Path(text_encoder_path))

    # Calculate latent dimensions
    stage1_h, stage1_w = height // 2 // 32, width // 2 // 32
    stage2_h, stage2_w = height // 32, width // 32
    latent_frames = 1 + (num_frames - 1) // 8
    stage1_token_count = latent_frames * stage1_h * stage1_w
    stage1_sigmas, stage2_sigmas = build_stage_sigma_schedules(
        num_inference_steps,
        stage1_token_count=stage1_token_count,
        use_ltx2_scheduler=True,
    )

    mx.random.seed(seed)

    # Load text encoder with audio embeddings
    print(f"{Colors.BLUE}📝 Loading text encoder...{Colors.RESET}")
    from mlx_video.models.ltx.text_encoder import LTX2TextEncoder

    text_encoder = LTX2TextEncoder()
    text_encoder.load(
        model_path=hf_model_path,
        text_encoder_path=text_encoder_path,
        use_unified=use_unified,
    )
    mx.eval(text_encoder.parameters())

    # Optionally enhance prompt
    if enhance_prompt:
        original_prompt = prompt
        try:
            if use_uncensored_enhancer:
                from mlx_video.models.ltx.enhance_prompt import enhance_with_model

                print(
                    f"{Colors.MAGENTA}✨ Enhancing prompt (uncensored)...{Colors.RESET}"
                )
                system_prompt = None
                if is_i2v:
                    from mlx_video.models.ltx.enhance_prompt import _load_system_prompt

                    system_prompt = _load_system_prompt("gemma_i2v_system_prompt.txt")
                prompt = enhance_with_model(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                    verbose=verbose,
                )
            else:
                print(f"{Colors.MAGENTA}✨ Enhancing prompt...{Colors.RESET}")
                prompt = text_encoder.enhance_t2v(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    verbose=verbose,
                )

            if not prompt or not str(prompt).strip():
                raise ValueError("Prompt enhancer returned empty output")

            # Structured token for host apps to capture final rewritten prompt.
            print(f"ENHANCED_PROMPT:{prompt}", file=sys.stderr, flush=True)
            print(
                f"{Colors.DIM}Enhanced: {prompt[:150]}{'...' if len(prompt) > 150 else ''}{Colors.RESET}"
            )
        except Exception as enhance_err:
            # Native fallback: preserve run stability even if enhancer fails.
            prompt = original_prompt
            print(
                f"{Colors.YELLOW}⚠️  Prompt enhancement failed; using original prompt: {enhance_err}{Colors.RESET}"
            )
            # Structured token for host apps (e.g. GUI bridge) to show clear fallback reason.
            print(
                f"ENHANCER_FALLBACK:{type(enhance_err).__name__}:{enhance_err}",
                file=sys.stderr,
                flush=True,
            )

    # Get both video and audio embeddings
    video_embeddings, audio_embeddings, text_attention_mask = text_encoder(
        prompt, max_length=1024
    )
    negative_prompt = (
        DEFAULT_NEGATIVE_PROMPT if negative_prompt is None else negative_prompt
    )
    cfg_enabled = cfg_scale > 1.0 and bool(str(negative_prompt).strip())
    if cfg_enabled:
        (
            video_embeddings_negative,
            audio_embeddings_negative,
            negative_attention_mask,
        ) = text_encoder(negative_prompt, max_length=1024)
    else:
        video_embeddings_negative = None
        audio_embeddings_negative = None
        negative_attention_mask = None

    model_dtype = video_embeddings.dtype  # bfloat16 from text encoder
    tensors_to_eval = [video_embeddings, audio_embeddings, text_attention_mask]
    if video_embeddings_negative is not None:
        tensors_to_eval.extend(
            [
                video_embeddings_negative,
                audio_embeddings_negative,
                negative_attention_mask,
            ]
        )
    mx.eval(*tensors_to_eval)

    del text_encoder
    mx.clear_cache()

    # Load transformer with AudioVideo config
    print(f"{Colors.BLUE}🤖 Loading transformer (A/V mode)...{Colors.RESET}")
    if use_unified:
        # Load from unified MLX model (weights already sanitized and converted)
        sanitized = load_unified_weights(model_path, "transformer.")
    else:
        # Load from HuggingFace format (needs sanitization)
        raw_weights = mx.load(str(model_path / "ltx-2-19b-distilled.safetensors"))
        sanitized = sanitize_transformer_weights(raw_weights)
        # Convert transformer weights to bfloat16 for memory efficiency
        sanitized = {
            k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v
            for k, v in sanitized.items()
        }

    caption_channels = 3840
    audio_caption_channels = 3840
    caption_proj_first = True
    caption_proj_second = True
    apply_gated_attention = False
    adaln_embedding_coefficient = 6
    embedded_transformer_cfg = {}
    embedded_scheduler_cfg = {}
    embedded_cfg_path = model_path / "embedded_config.json"
    if embedded_cfg_path.exists():
        try:
            with open(embedded_cfg_path, "r") as f:
                embedded_root_cfg = json.load(f)
            t_cfg = embedded_root_cfg.get("transformer", {})
            embedded_transformer_cfg = t_cfg
            embedded_scheduler_cfg = embedded_root_cfg.get("scheduler", {})
            caption_proj_first = t_cfg.get("caption_projection_first_linear", True)
            caption_proj_second = t_cfg.get("caption_projection_second_linear", True)
            apply_gated_attention = bool(t_cfg.get("apply_gated_attention", False))
            adaln_embedding_coefficient = 9 if apply_gated_attention else 6
            no_caption_proj = not caption_proj_first and not caption_proj_second
            if no_caption_proj:
                conn_heads = t_cfg.get("connector_num_attention_heads", 32)
                conn_head_dim = t_cfg.get("connector_attention_head_dim", 128)
                caption_channels = conn_heads * conn_head_dim
                audio_conn_heads = t_cfg.get("audio_connector_num_attention_heads", 32)
                audio_conn_head_dim = t_cfg.get(
                    "audio_connector_attention_head_dim", 64
                )
                audio_caption_channels = audio_conn_heads * audio_conn_head_dim
            else:
                caption_channels = t_cfg.get("caption_channels", caption_channels)
                audio_caption_channels = t_cfg.get(
                    "audio_caption_channels", audio_caption_channels
                )
        except (json.JSONDecodeError, OSError):
            pass

    config = LTXModelConfig(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=caption_channels,
        caption_projection_first_linear=caption_proj_first,
        caption_projection_second_linear=caption_proj_second,
        adaln_embedding_coefficient=adaln_embedding_coefficient,
        apply_gated_attention=apply_gated_attention,
        # Audio config
        audio_num_attention_heads=32,
        audio_attention_head_dim=64,
        audio_in_channels=AUDIO_LATENT_CHANNELS * AUDIO_MEL_BINS,  # 8 * 16 = 128
        audio_out_channels=AUDIO_LATENT_CHANNELS * AUDIO_MEL_BINS,
        audio_cross_attention_dim=2048,
        audio_caption_channels=audio_caption_channels,
        rope_type=LTXRopeType.SPLIT,
        double_precision_rope=True,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        audio_positional_embedding_max_pos=[20],
        use_middle_indices_grid=True,
        timestep_scale_multiplier=1000,
    )
    transformer = LTXModel(config)

    # Detect quantized model and selectively quantize layers that have quantized weights
    split_manifest = model_path / "split_model.json"
    if split_manifest.exists():
        try:
            with open(split_manifest, "r") as f:
                manifest = json.load(f)
            if manifest.get("quantized", False):
                q_bits = manifest.get("quantization_bits", 4)
                q_group = manifest.get("quantization_group_size", 64)
                quantized_paths = {
                    k.rsplit(".", 1)[0] for k in sanitized if k.endswith(".scales")
                }

                def _should_quantize(path: str, module: nn.Module) -> bool:
                    return isinstance(module, nn.Linear) and path in quantized_paths

                nn.quantize(
                    transformer,
                    group_size=q_group,
                    bits=q_bits,
                    class_predicate=_should_quantize,
                )
        except (json.JSONDecodeError, OSError):
            pass

    transformer.load_weights(list(sanitized.items()), strict=False)
    mx.eval(transformer.parameters())

    # Load VAE encoder and encode image for I2V conditioning
    stage1_image_latent = None
    stage2_image_latent = None
    if is_i2v:
        print(
            f"{Colors.BLUE}🖼️  Loading VAE encoder and encoding image...{Colors.RESET}"
        )
        vae_encoder = load_vae_encoder(
            (
                str(vae_model_path / "ltx-2-19b-distilled.safetensors")
                if (not use_unified or vae_model_path != model_path)
                else str(model_path)
            ),
            use_unified=use_unified and (vae_model_path == model_path),
        )
        mx.eval(vae_encoder.parameters())

        # Load and prepare image for stage 1 (half resolution)
        input_image = load_image(
            image, height=height // 2, width=width // 2, dtype=model_dtype
        )
        stage1_image_tensor = prepare_image_for_encoding(
            input_image, height // 2, width // 2, dtype=model_dtype
        )
        stage1_image_latent = vae_encoder(stage1_image_tensor)
        mx.eval(stage1_image_latent)

        # Load and prepare image for stage 2 (full resolution)
        input_image = load_image(image, height=height, width=width, dtype=model_dtype)
        stage2_image_tensor = prepare_image_for_encoding(
            input_image, height, width, dtype=model_dtype
        )
        stage2_image_latent = vae_encoder(stage2_image_tensor)
        mx.eval(stage2_image_latent)

        del vae_encoder
        mx.clear_cache()

    # Initialize latents
    print(
        f"{Colors.YELLOW}⚡ Stage 1: Generating at {width//2}x{height//2} ({len(stage1_sigmas) - 1} steps)...{Colors.RESET}"
    )
    mx.random.seed(seed)

    # Create position grids - MUST stay float32 for RoPE precision
    # bfloat16 positions cause quality degradation due to precision loss in sin/cos calculations
    video_positions = create_video_position_grid(
        1, latent_frames, stage1_h, stage1_w
    )  # float32
    audio_positions = create_audio_position_grid(1, audio_frames)  # float32
    mx.eval(video_positions, audio_positions)

    # Apply I2V conditioning for stage 1 if provided
    video_state1 = None
    video_latent_shape = (1, 128, latent_frames, stage1_h, stage1_w)
    if is_i2v and stage1_image_latent is not None:
        # PyTorch flow: create zeros -> apply conditioning -> apply noiser
        video_state1 = LatentState(
            latent=mx.zeros(video_latent_shape, dtype=model_dtype),
            clean_latent=mx.zeros(video_latent_shape, dtype=model_dtype),
            denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
        )
        conditioning = VideoConditionByLatentIndex(
            latent=stage1_image_latent,
            frame_idx=image_frame_idx,
            strength=image_strength,
        )
        video_state1 = apply_conditioning(video_state1, [conditioning])

        # Apply noiser: latent = noise * (mask * noise_scale) + latent * (1 - mask * noise_scale)
        noise = mx.random.normal(video_latent_shape).astype(model_dtype)
        noise_scale = mx.array(stage1_sigmas[0], dtype=model_dtype)  # 1.0
        scaled_mask = video_state1.denoise_mask * noise_scale
        video_state1 = LatentState(
            latent=noise * scaled_mask
            + video_state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
            clean_latent=video_state1.clean_latent,
            denoise_mask=video_state1.denoise_mask,
        )
        video_latents = video_state1.latent
        mx.eval(video_latents)
    else:
        # T2V: just use random noise
        video_latents = mx.random.normal(video_latent_shape).astype(model_dtype)
        mx.eval(video_latents)

    # Audio always uses pure noise (no I2V for audio)
    audio_latents = mx.random.normal(
        (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS)
    ).astype(model_dtype)
    mx.eval(audio_latents)
    # Stage 1 denoising
    video_latents, audio_latents = denoise_av(
        video_latents,
        audio_latents,
        video_positions,
        audio_positions,
        video_embeddings,
        audio_embeddings,
        video_embeddings_negative,
        audio_embeddings_negative,
        transformer,
        stage1_sigmas,
        verbose=verbose,
        video_state=video_state1,
        stage=1,
        cfg_scale=cfg_scale,
    )

    # Upsample video latents
    print(f"{Colors.MAGENTA}🔍 Upsampling video latents 2x...{Colors.RESET}")
    ltx_23_model = _is_ltx_23_model(model_repo, model_path)
    local_upsampler_filename = (
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        if ltx_23_model
        else "ltx-2-spatial-upscaler-x2-1.0.safetensors"
    )
    upsampler_fallback_candidates = (
        [("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")]
        if ltx_23_model
        else [("Lightricks/LTX-2", "ltx-2-spatial-upscaler-x2-1.0.safetensors")]
    )

    upsampler = load_upsampler(
        (
            str(hf_model_path / local_upsampler_filename)
            if not use_unified
            else str(model_path)
        ),
        use_unified=use_unified,
        fallback_candidates=upsampler_fallback_candidates,
    )
    mx.eval(upsampler.parameters())

    vae_decoder = load_vae_decoder(
        (
            str(vae_model_path / "ltx-2-19b-distilled.safetensors")
            if (not use_unified or vae_model_path != model_path)
            else str(model_path)
        ),
        timestep_conditioning=None,
        use_unified=use_unified and (vae_model_path == model_path),
    )

    video_latents = upsample_latents(
        video_latents, upsampler, vae_decoder.latents_mean, vae_decoder.latents_std
    )
    mx.eval(video_latents)

    del upsampler
    mx.clear_cache()

    # Stage 2: Refine at full resolution
    print(
        f"{Colors.YELLOW}⚡ Stage 2: Refining at {width}x{height} ({len(stage2_sigmas) - 1} steps)...{Colors.RESET}"
    )
    # Position grids stay float32 for RoPE precision
    video_positions = create_video_position_grid(
        1, latent_frames, stage2_h, stage2_w
    )  # float32
    mx.eval(video_positions)

    # Apply I2V conditioning for stage 2 if provided
    video_state2 = None
    if is_i2v and stage2_image_latent is not None:
        # PyTorch flow: start with upscaled latent -> apply conditioning -> apply noiser
        video_state2 = LatentState(
            latent=video_latents,  # Start with upscaled latent
            clean_latent=mx.zeros_like(video_latents),
            denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
        )
        conditioning = VideoConditionByLatentIndex(
            latent=stage2_image_latent,
            frame_idx=image_frame_idx,
            strength=image_strength,
        )
        video_state2 = apply_conditioning(video_state2, [conditioning])

        # Apply noiser: conditioned frames (mask=0) keep image latent, unconditioned get partial noise
        video_noise = mx.random.normal(video_latents.shape).astype(model_dtype)
        noise_scale = mx.array(stage2_sigmas[0], dtype=model_dtype)
        scaled_mask = video_state2.denoise_mask * noise_scale
        video_state2 = LatentState(
            latent=video_noise * scaled_mask
            + video_state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
            clean_latent=video_state2.clean_latent,
            denoise_mask=video_state2.denoise_mask,
        )
        video_latents = video_state2.latent
        mx.eval(video_latents)

        # Audio still gets noise (no I2V for audio)
        audio_noise = mx.random.normal(audio_latents.shape).astype(model_dtype)
        one_minus_scale = mx.array(1.0, dtype=model_dtype) - noise_scale
        audio_latents = audio_noise * noise_scale + audio_latents * one_minus_scale
        mx.eval(audio_latents)
    else:
        # T2V: add noise to all frames for refinement
        noise_scale = mx.array(stage2_sigmas[0], dtype=model_dtype)
        one_minus_scale = mx.array(1.0, dtype=model_dtype) - noise_scale
        video_noise = mx.random.normal(video_latents.shape).astype(model_dtype)
        audio_noise = mx.random.normal(audio_latents.shape).astype(model_dtype)
        video_latents = video_noise * noise_scale + video_latents * one_minus_scale
        audio_latents = audio_noise * noise_scale + audio_latents * one_minus_scale
        mx.eval(video_latents, audio_latents)

    video_latents, audio_latents = denoise_av(
        video_latents,
        audio_latents,
        video_positions,
        audio_positions,
        video_embeddings,
        audio_embeddings,
        video_embeddings_negative,
        audio_embeddings_negative,
        transformer,
        stage2_sigmas,
        verbose=verbose,
        video_state=video_state2,
        stage=2,
        cfg_scale=cfg_scale,
        use_gradient_estimation=True,
        ge_gamma=2.0,
    )

    del transformer
    mx.clear_cache()

    # Decode video with tiling
    print(f"{Colors.BLUE}🎞️  Decoding video...{Colors.RESET}")

    # Select tiling configuration
    if tiling == "none":
        tiling_config = None
    elif tiling == "auto":
        tiling_config = TilingConfig.auto(height, width, num_frames)
    elif tiling == "default":
        tiling_config = TilingConfig.default()
    elif tiling == "aggressive":
        tiling_config = TilingConfig.aggressive()
    elif tiling == "conservative":
        tiling_config = TilingConfig.conservative()
    elif tiling == "spatial":
        tiling_config = TilingConfig.spatial_only()
    elif tiling == "temporal":
        tiling_config = TilingConfig.temporal_only()
    else:
        print(
            f"{Colors.YELLOW}  Unknown tiling mode '{tiling}', using auto{Colors.RESET}"
        )
        tiling_config = TilingConfig.auto(height, width, num_frames)

    if tiling_config is not None:
        spatial_info = (
            f"{tiling_config.spatial_config.tile_size_in_pixels}px"
            if tiling_config.spatial_config
            else "none"
        )
        temporal_info = (
            f"{tiling_config.temporal_config.tile_size_in_frames}f"
            if tiling_config.temporal_config
            else "none"
        )
        print(
            f"{Colors.DIM}  Tiling ({tiling}): spatial={spatial_info}, temporal={temporal_info}{Colors.RESET}"
        )
        video = vae_decoder.decode_tiled(
            video_latents, tiling_config=tiling_config, debug=verbose
        )
    else:
        print(f"{Colors.DIM}  Tiling: disabled{Colors.RESET}")
        video = vae_decoder(video_latents)
    mx.eval(video)

    # Convert video to uint8 frames
    video = mx.squeeze(video, axis=0)
    video = mx.transpose(video, (1, 2, 3, 0))
    video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
    video = (video * 255).astype(mx.uint8)
    video_np = np.array(video)

    # Decode audio
    print(f"{Colors.BLUE}🔊 Decoding audio...{Colors.RESET}")
    audio_decoder = load_audio_decoder(model_path, use_unified=use_unified)
    vocoder = load_vocoder(model_path, use_unified=use_unified)
    audio_sample_rate = int(
        getattr(
            vocoder,
            "output_sampling_rate",
            getattr(vocoder, "output_sample_rate", AUDIO_SAMPLE_RATE),
        )
    )
    mx.eval(audio_decoder.parameters(), vocoder.parameters())

    mel_spectrogram = audio_decoder(audio_latents)
    mx.eval(mel_spectrogram)

    # Audio decoder output is already in vocoder format (B, C, T, F)
    audio_waveform = vocoder(mel_spectrogram)
    mx.eval(audio_waveform)

    audio_np = np.array(audio_waveform)
    if audio_np.ndim == 3:
        audio_np = audio_np[0]  # Remove batch dim

    del audio_decoder, vocoder, vae_decoder
    mx.clear_cache()

    # Save outputs
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save video (temporary without audio)
    temp_video_path = output_path.with_suffix(".temp.mp4")

    try:
        import cv2

        h, w = video_np.shape[1], video_np.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (w, h))
        for frame in video_np:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"{Colors.GREEN}✅ Video encoded{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}❌ Video encoding failed: {e}{Colors.RESET}")
        return None, None

    # Save audio (to temp file or final path)
    keep_audio_file = save_audio_separately or output_audio_path is not None
    if output_audio_path is not None:
        audio_path = Path(output_audio_path)
    elif save_audio_separately:
        audio_path = output_path.with_suffix(".wav")
    else:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        audio_path = Path(tmp)

    save_audio(audio_np, audio_path, audio_sample_rate)
    if keep_audio_file:
        print(f"{Colors.GREEN}✅ Saved audio to{Colors.RESET} {audio_path}")

    # Mux video and audio
    print(f"{Colors.BLUE}🎬 Combining video and audio...{Colors.RESET}")
    if mux_video_audio(temp_video_path, audio_path, output_path):
        print(f"{Colors.GREEN}✅ Saved video with audio to{Colors.RESET} {output_path}")
        temp_video_path.unlink()  # Remove temp file
        if not keep_audio_file:
            audio_path.unlink()  # Remove temp audio
    else:
        # Fallback: keep video without audio
        temp_video_path.rename(output_path)
        if not keep_audio_file:
            audio_path.unlink()  # Remove temp audio
        print(
            f"{Colors.YELLOW}⚠️  Saved video without audio to{Colors.RESET} {output_path}"
        )

    elapsed = time.time() - start_time
    print(
        f"{Colors.BOLD}{Colors.GREEN}🎉 Done! Generated in {elapsed:.1f}s{Colors.RESET}"
    )
    print(
        f"{Colors.BOLD}{Colors.GREEN}✨ Peak memory: {mx.get_peak_memory() / (1024 ** 3):.2f}GB{Colors.RESET}"
    )

    return video_np, audio_np


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos with synchronized audio using MLX LTX-2 (T2V and I2V)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-to-Video with Audio (T2V+Audio)
  python -m mlx_video.generate_av --prompt "Ocean waves crashing on a beach"
  python -m mlx_video.generate_av --prompt "A jazz band playing" --enhance-prompt
  python -m mlx_video.generate_av --prompt "..." --output my_video.mp4 --output-audio my_audio.wav

  # Image-to-Video with Audio (I2V+Audio)
  python -m mlx_video.generate_av --prompt "A person dancing" --image photo.jpg
  python -m mlx_video.generate_av --prompt "Waves crashing" --image beach.png --image-strength 0.8
        """,
    )

    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Text description of the video/audio to generate",
    )
    parser.add_argument(
        "--height",
        "-H",
        type=int,
        default=512,
        help="Output video height (default: 512)",
    )
    parser.add_argument(
        "--width", "-W", type=int, default=512, help="Output video width (default: 512)"
    )
    parser.add_argument(
        "--num-frames",
        "-n",
        type=int,
        default=65,
        help="Number of frames (default: 65)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--fps", type=int, default=24, help="Frames per second (default: 24)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output_av.mp4",
        help="Output video path (default: output_av.mp4)",
    )
    parser.add_argument(
        "--output-audio",
        type=str,
        default=None,
        help="Output audio path (default: same as video with .wav)",
    )
    parser.add_argument(
        "--save-audio-separately",
        action="store_true",
        help="Keep the .wav audio file alongside the video (default: off, audio only in mp4)",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="notapalindrome/ltx2-mlx-av",
        help="Model repository (default: notapalindrome/ltx2-mlx-av, ~42GB unified MLX)",
    )
    parser.add_argument(
        "--text-encoder-repo", type=str, default=None, help="Text encoder repository"
    )
    parser.add_argument(
        "--num-inference-steps",
        "--steps",
        dest="num_inference_steps",
        type=int,
        default=30,
        help="Total denoising steps across both stages (default: 30, matches app)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for CFG guidance (default: official LTX negative prompt)",
    )
    parser.add_argument(
        "--cfg-scale",
        "--guidance-scale",
        dest="cfg_scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale (default: 3.0, set 1.0 to disable)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--enhance-prompt", action="store_true", help="Enhance prompt using Gemma"
    )
    parser.add_argument(
        "--use-uncensored-enhancer",
        action="store_true",
        help="Use uncensored Gemma 12B for prompt enhancement (avoids content filters)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Max tokens for prompt enhancement"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for prompt enhancement",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to conditioning image for I2V (Image-to-Video) generation",
    )
    parser.add_argument(
        "--image-strength",
        type=float,
        default=1.0,
        help="Conditioning strength for I2V (1.0 = full denoise, 0.0 = keep original, default: 1.0)",
    )
    parser.add_argument(
        "--image-frame-idx",
        type=int,
        default=0,
        help="Frame index to condition for I2V (0 = first frame, default: 0)",
    )
    parser.add_argument(
        "--tiling",
        type=str,
        default="auto",
        choices=[
            "auto",
            "none",
            "default",
            "aggressive",
            "conservative",
            "spatial",
            "temporal",
        ],
        help="Tiling mode for VAE decoding (default: auto). "
        "auto=based on size, none=disabled, default=512px/64f, "
        "aggressive=256px/32f (lowest memory), conservative=768px/96f",
    )

    args = parser.parse_args()

    generate_video_with_audio(
        model_repo=args.model_repo,
        text_encoder_repo=args.text_encoder_repo,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        negative_prompt=args.negative_prompt,
        cfg_scale=args.cfg_scale,
        output_path=args.output_path,
        output_audio_path=args.output_audio,
        save_audio_separately=args.save_audio_separately,
        verbose=args.verbose,
        enhance_prompt=args.enhance_prompt,
        use_uncensored_enhancer=args.use_uncensored_enhancer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        image=args.image,
        image_strength=args.image_strength,
        image_frame_idx=args.image_frame_idx,
        tiling=args.tiling,
    )


if __name__ == "__main__":
    main()
