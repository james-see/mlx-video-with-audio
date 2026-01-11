import mlx.core as mx
import numpy as np
from pathlib import Path
from PIL import Image

from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType
from mlx_video.models.ltx.ltx import LTXModel
from mlx_video.models.ltx.transformer import Modality
from mlx_video.convert import sanitize_transformer_weights
from mlx_video.generate import create_position_grid
from mlx_video.utils import to_denoised
from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder
from mlx_video.models.ltx.upsampler import LatentUpsampler, load_upsampler, upsample_latents

# Paths
from huggingface_hub import snapshot_download
from pathlib import Path
import os

LTX2_REPO = "Lightricks/LTX-2"

def get_ltx2_cache_dir():
    # Try to get local cache (local_only), will not download files
    try:
        ref_path = snapshot_download(
            repo_id=LTX2_REPO,
            local_files_only=True,
            allow_patterns=["*"],
            ignore_patterns=[],
            # leave as default revision and cache_dir, only local
        )
        return ref_path
    except Exception:
        # If not present locally, download from hub
        return snapshot_download(
            repo_id=LTX2_REPO,
            local_files_only=False,
            resume_download=True,
            allow_patterns=["*.safetensors", "*.json"],
            ignore_patterns=[]
        )

LTX2_PATH = Path(get_ltx2_cache_dir())
MODEL_PATH = str(LTX2_PATH / 'ltx-2-19b-distilled.safetensors')
UPSAMPLER_PATH = str(LTX2_PATH / 'ltx-2-spatial-upscaler-x2-1.0.safetensors')
TEXT_ENCODER_PATH = str(LTX2_PATH / 'text_encoder')
TOKENIZER_PATH = str(LTX2_PATH / 'tokenizer')

# Distilled sigma schedules (from PyTorch)
STAGE_1_SIGMA_SCHEDULE = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_SIGMA_SCHEDULE = [0.909375, 0.725, 0.421875, 0.0]  # Refinement steps


def denoise_loop(
    latents: mx.array,
    positions: mx.array,
    text_embeddings: mx.array,
    transformer: LTXModel,
    sigma_schedule: list,
    stage_name: str = "Stage",
    negative_embeddings: mx.array = None,
    cfg_scale: float = 1.0,
) -> mx.array:
    """Run denoising loop for given sigma schedule.

    Args:
        latents: Noisy latent tensor
        positions: Position embeddings
        text_embeddings: Positive prompt embeddings
        transformer: The transformer model
        sigma_schedule: List of sigma values for each step
        stage_name: Name for logging
        negative_embeddings: Negative prompt embeddings for CFG (optional)
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
    """
    use_cfg = negative_embeddings is not None and cfg_scale > 1.0

    for i in range(len(sigma_schedule) - 1):
        sigma = sigma_schedule[i]
        sigma_next = sigma_schedule[i + 1]

        print(f"  {stage_name} step {i+1}/{len(sigma_schedule)-1}: sigma={sigma:.4f} -> {sigma_next:.4f}")

        b, c, f, h, w = latents.shape
        latents_flat = mx.reshape(latents, (b, c, -1))
        latents_flat = mx.transpose(latents_flat, (0, 2, 1))

        timesteps = mx.full((1,), sigma)

        # Positive (conditioned) prediction
        video_modality = Modality(
            latent=latents_flat,
            timesteps=timesteps,
            positions=positions,
            context=text_embeddings,
            context_mask=None,
            enabled=True,
        )

        vx_cond, _ = transformer(video=video_modality, audio=None)
        mx.eval(vx_cond)

        if use_cfg:
            # Negative (unconditioned) prediction
            video_modality_neg = Modality(
                latent=latents_flat,
                timesteps=timesteps,
                positions=positions,
                context=negative_embeddings,
                context_mask=None,
                enabled=True,
            )
            vx_uncond, _ = transformer(video=video_modality_neg, audio=None)
            mx.eval(vx_uncond)

            # CFG: output = uncond + cfg_scale * (cond - uncond)
            vx = vx_uncond + cfg_scale * (vx_cond - vx_uncond)
        else:
            vx = vx_cond

        vx_reshaped = mx.transpose(vx, (0, 2, 1))
        vx_reshaped = mx.reshape(vx_reshaped, (b, c, f, h, w))

        # Debug: Print velocity stats
        vx_np = np.array(vx_reshaped)
        print(f"    Velocity: min={vx_np.min():.4f}, max={vx_np.max():.4f}, mean={vx_np.mean():.4f}")

        # Get denoised prediction: x_0 = x_t - sigma * velocity
        denoised = to_denoised(latents, vx_reshaped, sigma)
        mx.eval(denoised)

        # Debug: Print denoised stats
        denoised_np = np.array(denoised)
        print(f"    Denoised: min={denoised_np.min():.4f}, max={denoised_np.max():.4f}, mean={denoised_np.mean():.4f}")

        # Euler step: x_next = x_0 + sigma_next * (x_t - x_0) / sigma
        if sigma_next > 0:
            velocity = (latents - denoised) / sigma
            latents = denoised + sigma_next * velocity
        else:
            latents = denoised
        mx.eval(latents)

        # Debug: Print latents after step
        latents_np = np.array(latents)
        print(f"    Latents after step: min={latents_np.min():.4f}, max={latents_np.max():.4f}, mean={latents_np.mean():.4f}")


    return latents


def main():
    print("="*60)
    print("MLX LTX-2 Video Generation (Two-Stage)")
    print("="*60)

    # Config - same as PyTorch reference
    prompt = "A beautiful woman with flowing dark hair stands on a tropical beach at golden hour, gentle waves lapping at her feet, she turns and smiles at the camera, warm sunlight illuminating her face, palm trees swaying in the background, cinematic lighting, photorealistic"
    negative_prompt = ""  # PyTorch script doesn't use negative prompt
    cfg_scale = 1.0  # No CFG in the distilled pipeline
    height, width, num_frames = 512, 512, 500 # Must be divisible by 64 for two-stage
    seed = 123

    # Stage 1: Half resolution
    stage1_height = height // 2
    stage1_width = width // 2
    stage1_latent_height = stage1_height // 32
    stage1_latent_width = stage1_width // 32
    latent_frames = 1 + (num_frames - 1) // 8

    # Stage 2: Full resolution
    latent_height = height // 32
    latent_width = width // 32

    print(f"\nConfig:")
    print(f"  Prompt: {prompt}")
    print(f"  Negative prompt: '{negative_prompt}'")
    print(f"  CFG scale: {cfg_scale}")
    print(f"  Final resolution: {width}x{height}, {num_frames} frames")
    print(f"  Stage 1: {stage1_width}x{stage1_height} -> latent {stage1_latent_width}x{stage1_latent_height}")
    print(f"  Stage 2: {width}x{height} -> latent {latent_width}x{latent_height}")
    print(f"  Seed: {seed}")

    mx.random.seed(seed)

    # Load text encoder
    print("\nLoading text encoder...")
    from mlx_video.models.ltx.text_encoder import LTX2TextEncoder

    text_encoder = LTX2TextEncoder(model_path=str(LTX2_PATH))
    text_encoder.load(str(LTX2_PATH))
    mx.eval(text_encoder.parameters())

    # Encode positive prompt
    print("Encoding text...")
    text_embeddings, attention_mask = text_encoder(prompt)
    mx.eval(text_embeddings)
    print(f"  Positive embeddings: {text_embeddings.shape}")

    # Encode negative prompt for CFG
    negative_embeddings, _ = text_encoder(negative_prompt)
    mx.eval(negative_embeddings)
    print(f"  Negative embeddings: {negative_embeddings.shape}")

    # Free text encoder memory
    del text_encoder
    mx.clear_cache()

    # Load transformer
    print("\nLoading transformer...")
    raw_weights = mx.load(MODEL_PATH)
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
    print("  Transformer loaded!")

    # ========================================
    # Stage 1: Generate at half resolution
    # ========================================
    print("\n" + "="*60)
    print("Stage 1: Generating at half resolution")
    print("="*60)

    mx.random.seed(seed)
    latents = mx.random.normal((1, 128, latent_frames, stage1_latent_height, stage1_latent_width))
    mx.eval(latents)
    print(f"  Initial latents: {latents.shape}")

    positions = create_position_grid(1, latent_frames, stage1_latent_height, stage1_latent_width)
    mx.eval(positions)

    latents = denoise_loop(
        latents=latents,
        positions=positions,
        text_embeddings=text_embeddings,
        transformer=transformer,
        sigma_schedule=STAGE_1_SIGMA_SCHEDULE,
        stage_name="Stage 1",
        negative_embeddings=negative_embeddings,
        cfg_scale=cfg_scale,
    )

    print(f"\nStage 1 latents: {latents.shape}")
    latents_np = np.array(latents)
    print(f"  Stats: min={latents_np.min():.4f}, max={latents_np.max():.4f}, mean={latents_np.mean():.4f}")

    # ========================================
    # Upsample latents 2x
    # ========================================
    print("\n" + "="*60)
    print("Upsampling latents 2x")
    print("="*60)

    # Load upsampler
    print("  Loading spatial upsampler...")
    upsampler = load_upsampler(UPSAMPLER_PATH)
    mx.eval(upsampler.parameters())

    # Load latent statistics for normalization
    vae_decoder = load_vae_decoder(MODEL_PATH, timestep_conditioning=True)
    # EXPERIMENT: Disable VAE decode noise for sharper output
    # vae_decoder.decode_noise_scale = 0.0
    # print(f"  VAE decode_noise_scale set to {vae_decoder.decode_noise_scale}")
    latent_mean = vae_decoder.latents_mean
    latent_std = vae_decoder.latents_std

    # Upsample
    print("  Upsampling...")
    latents = upsample_latents(latents, upsampler, latent_mean, latent_std, debug=False)
    mx.eval(latents)
    print(f"  Upsampled latents: {latents.shape}")

    # Free upsampler memory
    del upsampler
    mx.clear_cache()

    # ========================================
    # Stage 2: Refine at full resolution
    # ========================================
    print("\n" + "="*60)
    print("Stage 2: Refining at full resolution")
    print("="*60)

    # Debug: Print upsampled latent stats before adding noise
    latents_np = np.array(latents)
    print(f"  Upsampled latents (before noise): min={latents_np.min():.4f}, max={latents_np.max():.4f}, mean={latents_np.mean():.4f}")

    # Create new position grid for full resolution
    positions = create_position_grid(1, latent_frames, latent_height, latent_width)
    mx.eval(positions)

    # Add noise at initial sigma for stage 2
    # PyTorch uses interpolation: noisy = noise * scale + clean * (1 - scale)
    # NOT addition: noisy = clean + scale * noise
    noise_scale = STAGE_2_SIGMA_SCHEDULE[0]
    noise = mx.random.normal(latents.shape)
    latents = noise * noise_scale + latents * (1 - noise_scale)
    mx.eval(latents)

    # Debug: Print latents after adding noise
    latents_np = np.array(latents)
    print(f"  After adding noise (sigma={noise_scale}): min={latents_np.min():.4f}, max={latents_np.max():.4f}, mean={latents_np.mean():.4f}")

    latents = denoise_loop(
        latents=latents,
        positions=positions,
        text_embeddings=text_embeddings,
        transformer=transformer,
        sigma_schedule=STAGE_2_SIGMA_SCHEDULE,
        stage_name="Stage 2",
    )

    print(f"\nFinal latents: {latents.shape}")
    latents_np = np.array(latents)
    print(f"  Stats: min={latents_np.min():.4f}, max={latents_np.max():.4f}, mean={latents_np.mean():.4f}")

    # Save latents for PyTorch comparison
    np.save("mlx_final_latents.npy", latents_np)
    print("  Saved latents to mlx_final_latents.npy")

    # Free transformer memory
    del transformer
    mx.clear_cache()

    # ========================================
    # Decode to video
    # ========================================
    print("\n" + "="*60)
    print("Decoding with VAE")
    print("="*60)

    # Decode latents to video
    video = vae_decoder(latents, debug=True)
    mx.eval(video)
    print(f"  Video shape: {video.shape}")

    # Convert to frames
    video = mx.squeeze(video, axis=0)  # (C, F, H, W)

    # Debug: check raw RGB values before conversion
    video_raw = np.array(video)
    print(f"  Raw video per-channel means: R={video_raw[0].mean():.4f}, G={video_raw[1].mean():.4f}, B={video_raw[2].mean():.4f}")
    print(f"  Raw video range: [{video_raw.min():.4f}, {video_raw.max():.4f}]")

    video = mx.transpose(video, (1, 2, 3, 0))  # (F, H, W, C)
    video = (video + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    video = mx.clip(video, 0.0, 1.0)
    video = (video * 255).astype(mx.uint8)
    video_np = np.array(video)

    print(f"  Converted video RGB means: R={video_np[:,:,:,0].mean():.1f}, G={video_np[:,:,:,1].mean():.1f}, B={video_np[:,:,:,2].mean():.1f}")

    # Save first frame
    output_path = Path("mlx_output_frame0_2.png")
    Image.fromarray(video_np[0]).save(output_path)
    print(f"\nSaved first frame to {output_path}")

    # Save video
    try:
        import imageio
        video_path = "mlx_output_video_2.mp4"
        imageio.mimwrite(video_path, video_np, fps=24, codec='libx264')
        print(f"Saved video to {video_path}")
    except Exception as e:
        print(f"Could not save video: {e}")

    print("\nDone!")


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
