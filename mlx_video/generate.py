import argparse
import time
from pathlib import Path
from typing import Optional, List, Tuple

import mlx.core as mx
import numpy as np
from PIL import Image
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
from mlx_video.convert import sanitize_transformer_weights, sanitize_vae_encoder_weights
from mlx_video.utils import to_denoised, load_image, prepare_image_for_encoding
from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder
from mlx_video.models.ltx.video_vae.encoder import load_vae_encoder
from mlx_video.models.ltx.upsampler import load_upsampler, upsample_latents
from mlx_video.conditioning import VideoConditionByLatentIndex, apply_conditioning
from mlx_video.conditioning.latent import LatentState, create_initial_state, apply_denoise_mask, add_noise_with_state

from mlx_video.utils import get_model_path


# Distilled sigma schedules
STAGE_1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]


def create_position_grid(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    temporal_scale: int = 8,
    spatial_scale: int = 32,
    fps: float = 24.0,
    causal_fix: bool = True,
) -> mx.array:
    """Create position grid for RoPE in pixel space.

    Args:
        batch_size: Batch size
        num_frames: Number of frames (latent)
        height: Height (latent)
        width: Width (latent)
        temporal_scale: VAE temporal scale factor (default 8)
        spatial_scale: VAE spatial scale factor (default 32)
        fps: Frames per second (default 24.0)
        causal_fix: Apply causal fix for first frame (default True)

    Returns:
        Position grid of shape (B, 3, num_patches, 2) in pixel space
        where dim 2 is [start, end) bounds for each patch
    """
    # Patch size is (1, 1, 1) for LTX-2 - no spatial patching
    patch_size_t, patch_size_h, patch_size_w = 1, 1, 1

    # Generate grid coordinates for each dimension (frame, height, width)
    t_coords = np.arange(0, num_frames, patch_size_t)
    h_coords = np.arange(0, height, patch_size_h)
    w_coords = np.arange(0, width, patch_size_w)

    # Create meshgrid with indexing='ij' for (frame, height, width) order
    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing='ij')

    # Stack to get shape (3, grid_t, grid_h, grid_w)
    patch_starts = np.stack([t_grid, h_grid, w_grid], axis=0)

    # Calculate end coordinates (start + patch_size)
    patch_size_delta = np.array([patch_size_t, patch_size_h, patch_size_w]).reshape(3, 1, 1, 1)
    patch_ends = patch_starts + patch_size_delta

    # Stack start and end: shape (3, grid_t, grid_h, grid_w, 2)
    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)

    # Flatten spatial/temporal dims: (3, num_patches, 2)
    num_patches = num_frames * height * width
    latent_coords = latent_coords.reshape(3, num_patches, 2)

    # Broadcast to batch: (batch, 3, num_patches, 2)
    latent_coords = np.tile(latent_coords[np.newaxis, ...], (batch_size, 1, 1, 1))

    # Convert latent coords to pixel coords by scaling with VAE factors
    scale_factors = np.array([temporal_scale, spatial_scale, spatial_scale]).reshape(1, 3, 1, 1)
    pixel_coords = (latent_coords * scale_factors).astype(np.float32)

    # Apply causal fix for first frame temporal axis
    if causal_fix:
        # VAE temporal stride for first frame is 1 instead of temporal_scale
        pixel_coords[:, 0, :, :] = np.clip(
            pixel_coords[:, 0, :, :] + 1 - temporal_scale,
            a_min=0,
            a_max=None
        )

    # Convert temporal to time in seconds by dividing by fps
    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps

    return mx.array(pixel_coords, dtype=mx.float32)


def denoise(
    latents: mx.array,
    positions: mx.array,
    text_embeddings: mx.array,
    transformer: LTXModel,
    sigmas: list,
    verbose: bool = True,
    state: Optional[LatentState] = None,
) -> mx.array:
    """Run denoising loop with optional conditioning.

    Args:
        latents: Noisy latent tensor (B, C, F, H, W)
        positions: Position embeddings
        text_embeddings: Text conditioning embeddings
        transformer: LTX model
        sigmas: List of sigma values for denoising schedule
        verbose: Whether to show progress bar
        state: Optional LatentState for I2V conditioning

    Returns:
        Denoised latent tensor
    """
    # If state is provided, use its latent (which may have conditioning applied)
    if state is not None:
        latents = state.latent

    for i in tqdm(range(len(sigmas) - 1), desc="Denoising", disable=not verbose):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]

        b, c, f, h, w = latents.shape
        num_tokens = f * h * w
        latents_flat = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))

        # Compute per-token timesteps
        # For I2V: conditioned tokens get timestep=0 (mask=0), unconditioned get timestep=sigma (mask=1)
        if state is not None:
            # Reshape denoise_mask from (B, 1, F, 1, 1) to (B, num_tokens)
            denoise_mask_flat = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
            denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
            denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_tokens))
            # Per-token timesteps: sigma * mask
            timesteps = sigma * denoise_mask_flat
        else:
            # All tokens get the same timestep
            timesteps = mx.full((b, num_tokens), sigma)

        video_modality = Modality(
            latent=latents_flat,
            timesteps=timesteps,
            positions=positions,
            context=text_embeddings,
            context_mask=None,
            enabled=True,
        )

        velocity, _ = transformer(video=video_modality, audio=None)
        mx.eval(velocity)

        velocity = mx.reshape(mx.transpose(velocity, (0, 2, 1)), (b, c, f, h, w))
        denoised = to_denoised(latents, velocity, sigma)

        # Apply conditioning mask if state is provided
        if state is not None:
            denoised = apply_denoise_mask(denoised, state.clean_latent, state.denoise_mask)

        mx.eval(denoised)

        if sigma_next > 0:
            latents = denoised + sigma_next * (latents - denoised) / sigma
        else:
            latents = denoised
        mx.eval(latents)

    return latents


def generate_video(
    model_repo: str,
    text_encoder_repo: str,
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_frames: int = 33,
    seed: int = 42,
    fps: int = 24,
    output_path: str = "output.mp4",
    save_frames: bool = False,
    verbose: bool = True,
    enhance_prompt: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.7,
    image: Optional[str] = None,
    image_strength: float = 1.0,
    image_frame_idx: int = 0,
):
    """Generate video from text prompt, optionally conditioned on an image.

    Args:
        model_repo: Model repository ID
        text_encoder_repo: Text encoder repository ID
        prompt: Text description of the video to generate
        height: Output video height (must be divisible by 64)
        width: Output video width (must be divisible by 64)
        num_frames: Number of frames (must be 1 + 8*k, e.g., 33, 65, 97)
        seed: Random seed for reproducibility
        fps: Frames per second for output video
        output_path: Path to save the output video
        save_frames: Whether to save individual frames as images
        verbose: Whether to print progress
        enhance_prompt: Whether to enhance prompt using Gemma
        max_tokens: Max tokens for prompt enhancement
        temperature: Temperature for prompt enhancement
        image: Path to conditioning image for I2V (Image-to-Video)
        image_strength: Conditioning strength (1.0 = full denoise, 0.0 = keep original)
        image_frame_idx: Frame index to condition (0 = first frame)
    """
    start_time = time.time()

    # Validate dimensions
    assert height % 64 == 0, f"Height must be divisible by 64, got {height}"
    assert width % 64 == 0, f"Width must be divisible by 64, got {width}"
    
    if num_frames % 8 != 1:
        adjusted_num_frames = round((num_frames - 1) / 8) * 8 + 1
        print(f"{Colors.YELLOW}⚠️  Number of frames must be 1 + 8*k. Using nearest valid value: {adjusted_num_frames}{Colors.RESET}")
        num_frames = adjusted_num_frames


    is_i2v = image is not None
    mode_str = "I2V" if is_i2v else "T2V"
    print(f"{Colors.BOLD}{Colors.CYAN}🎬 [{mode_str}] Generating {width}x{height} video with {num_frames} frames{Colors.RESET}")
    print(f"{Colors.DIM}Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}{Colors.RESET}")
    if is_i2v:
        print(f"{Colors.DIM}Image: {image} (strength={image_strength}, frame={image_frame_idx}){Colors.RESET}")

    # Get model path
    model_path = get_model_path(model_repo)
    text_encoder_path = model_path if text_encoder_repo is None else get_model_path(text_encoder_repo)

    # Calculate latent dimensions
    stage1_h, stage1_w = height // 2 // 32, width // 2 // 32
    stage2_h, stage2_w = height // 32, width // 32
    latent_frames = 1 + (num_frames - 1) // 8

    mx.random.seed(seed)

    # Load text encoder
    print(f"{Colors.BLUE}📝 Loading text encoder...{Colors.RESET}")
    from mlx_video.models.ltx.text_encoder import LTX2TextEncoder
    text_encoder = LTX2TextEncoder()
    text_encoder.load(model_path=model_path, text_encoder_path=text_encoder_path)
    mx.eval(text_encoder.parameters())

    # Optionally enhance the prompt
    if enhance_prompt:
        print(f"{Colors.MAGENTA}✨ Enhancing prompt...{Colors.RESET}")
        prompt = text_encoder.enhance_t2v(prompt, max_tokens=max_tokens, temperature=temperature, seed=seed, verbose=verbose)
        print(f"{Colors.DIM}Enhanced: {prompt[:150]}{'...' if len(prompt) > 150 else ''}{Colors.RESET}")

    text_embeddings, _ = text_encoder(prompt, return_audio_embeddings=False)
    mx.eval(text_embeddings)

    del text_encoder
    mx.clear_cache()

    # Load transformer
    print(f"{Colors.BLUE}🤖 Loading transformer...{Colors.RESET}")
    raw_weights = mx.load(str(model_path / 'ltx-2-19b-distilled.safetensors'))
    sanitized = sanitize_transformer_weights(raw_weights)

    config = LTXModelConfig(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
        rope_type=LTXRopeType.SPLIT,
        double_precision_rope=True,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        use_middle_indices_grid=True,
        timestep_scale_multiplier=1000,
    )

    transformer = LTXModel(config)
    transformer.load_weights(list(sanitized.items()), strict=False)
    mx.eval(transformer.parameters())

    # Load VAE encoder and encode image for I2V conditioning
    stage1_image_latent = None
    stage2_image_latent = None
    if is_i2v:
        print(f"{Colors.BLUE}🖼️  Loading VAE encoder and encoding image...{Colors.RESET}")
        vae_encoder = load_vae_encoder(str(model_path / 'ltx-2-19b-distilled.safetensors'))
        mx.eval(vae_encoder.parameters())

        # Load and prepare image for stage 1 (half resolution)
        input_image = load_image(image, height=height // 2, width=width // 2)
        stage1_image_tensor = prepare_image_for_encoding(input_image, height // 2, width // 2)
        stage1_image_latent = vae_encoder(stage1_image_tensor)
        mx.eval(stage1_image_latent)
        print(f"  Stage 1 image latent: {stage1_image_latent.shape}")

        # Load and prepare image for stage 2 (full resolution)
        input_image = load_image(image, height=height, width=width)
        stage2_image_tensor = prepare_image_for_encoding(input_image, height, width)
        stage2_image_latent = vae_encoder(stage2_image_tensor)
        mx.eval(stage2_image_latent)
        print(f"  Stage 2 image latent: {stage2_image_latent.shape}")

        del vae_encoder
        mx.clear_cache()

    # Stage 1: Generate at half resolution
    print(f"{Colors.YELLOW}⚡ Stage 1: Generating at {width//2}x{height//2} (8 steps)...{Colors.RESET}")
    mx.random.seed(seed)
    latents = mx.random.normal((1, 128, latent_frames, stage1_h, stage1_w))
    mx.eval(latents)

    positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
    mx.eval(positions)

    # Apply I2V conditioning if provided
    state1 = None
    if is_i2v and stage1_image_latent is not None:
        # Create state with conditioning
        state1 = LatentState(
            latent=latents,
            clean_latent=mx.zeros_like(latents),
            denoise_mask=mx.ones((1, 1, latent_frames, 1, 1)),
        )
        conditioning = VideoConditionByLatentIndex(
            latent=stage1_image_latent,
            frame_idx=image_frame_idx,
            strength=image_strength,
        )
        state1 = apply_conditioning(state1, [conditioning])
        latents = state1.latent
        mx.eval(latents)

    latents = denoise(latents, positions, text_embeddings, transformer, STAGE_1_SIGMAS, verbose=verbose, state=state1)

    # Upsample latents
    print(f"{Colors.MAGENTA}🔍 Upsampling latents 2x...{Colors.RESET}")
    upsampler = load_upsampler(str(model_path / 'ltx-2-spatial-upscaler-x2-1.0.safetensors'))
    mx.eval(upsampler.parameters())

    vae_decoder = load_vae_decoder(
        str(model_path / 'ltx-2-19b-distilled.safetensors'),
        timestep_conditioning=None  # Auto-detect from model metadata
    )

    latents = upsample_latents(latents, upsampler, vae_decoder.latents_mean, vae_decoder.latents_std)
    mx.eval(latents)

    del upsampler
    mx.clear_cache()

    # Stage 2: Refine at full resolution
    print(f"{Colors.YELLOW}⚡ Stage 2: Refining at {width}x{height} (3 steps)...{Colors.RESET}")
    positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
    mx.eval(positions)

    # Add noise for refinement
    noise_scale = STAGE_2_SIGMAS[0]
    noise = mx.random.normal(latents.shape)
    latents = noise * noise_scale + latents * (1 - noise_scale)
    mx.eval(latents)

    # Apply I2V conditioning for stage 2 if provided
    state2 = None
    if is_i2v and stage2_image_latent is not None:
        state2 = LatentState(
            latent=latents,
            clean_latent=mx.zeros_like(latents),
            denoise_mask=mx.ones((1, 1, latent_frames, 1, 1)),
        )
        conditioning = VideoConditionByLatentIndex(
            latent=stage2_image_latent,
            frame_idx=image_frame_idx,
            strength=image_strength,
        )
        state2 = apply_conditioning(state2, [conditioning])
        latents = state2.latent
        mx.eval(latents)

    latents = denoise(latents, positions, text_embeddings, transformer, STAGE_2_SIGMAS, verbose=verbose, state=state2)

    del transformer
    mx.clear_cache()

    # Decode to video
    print(f"{Colors.BLUE}🎞️  Decoding video...{Colors.RESET}")
    video = vae_decoder(latents)
    mx.eval(video)
    mx.clear_cache()

    # Convert to uint8 frames
    video = mx.squeeze(video, axis=0)  # (C, F, H, W)
    video = mx.transpose(video, (1, 2, 3, 0))  # (F, H, W, C)
    video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
    video = (video * 255).astype(mx.uint8)
    video_np = np.array(video)

    # Save outputs
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
        height, width = video_np.shape[1], video_np.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        for frame in video_np:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"{Colors.GREEN}✅ Saved video to{Colors.RESET} {output_path}")
    except Exception as e:
        print(f"{Colors.RED}❌ Could not save video: {e}{Colors.RESET}")

    if save_frames:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(video_np):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
        print(f"{Colors.GREEN}✅ Saved {len(video_np)} frames to {frames_dir}{Colors.RESET}")

    elapsed = time.time() - start_time
    print(f"{Colors.BOLD}{Colors.GREEN}🎉 Done! Generated in {elapsed:.1f}s ({elapsed/num_frames:.2f}s/frame){Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}✨ Peak memory: {mx.get_peak_memory() / (1024 ** 3):.2f}GB{Colors.RESET}")

    return video_np


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos with MLX LTX-2 (T2V and I2V)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-to-Video (T2V)
  python -m mlx_video.generate --prompt "A cat walking on grass"
  python -m mlx_video.generate --prompt "Ocean waves at sunset" --height 768 --width 768
  python -m mlx_video.generate --prompt "..." --num-frames 65 --seed 123 --output my_video.mp4

  # Image-to-Video (I2V)
  python -m mlx_video.generate --prompt "A person dancing" --image photo.jpg
  python -m mlx_video.generate --prompt "Waves crashing" --image beach.png --image-strength 0.8
        """
    )

    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Text description of the video to generate"
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=512,
        help="Output video height (default: 512, must be divisible by 32)"
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=512,
        help="Output video width (default: 512, must be divisible by 32)"
    )
    parser.add_argument(
        "--num-frames", "-n",
        type=int,
        default=100,
        help="Number of frames (default: 100)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for output video (default: 24)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output.mp4",
        help="Output video path (default: output.mp4)"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save individual frames as images"
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="Lightricks/LTX-2",
        help="Model repository to use (default: Lightricks/LTX-2)"
    )
    parser.add_argument(
        "--text-encoder-repo",
        type=str,
        default=None,
        help="Text encoder repository to use (default: None)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Enhance the prompt using Gemma before generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for prompt enhancement (default: 0.7)"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to conditioning image for I2V (Image-to-Video) generation"
    )
    parser.add_argument(
        "--image-strength",
        type=float,
        default=1.0,
        help="Conditioning strength for I2V (1.0 = full denoise, 0.0 = keep original, default: 1.0)"
    )
    parser.add_argument(
        "--image-frame-idx",
        type=int,
        default=0,
        help="Frame index to condition for I2V (0 = first frame, default: 0)"
    )
    args = parser.parse_args()

    generate_video(
        **vars(args)
    )


if __name__ == "__main__":
    main()
