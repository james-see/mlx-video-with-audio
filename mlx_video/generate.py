from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Iterator, Union

import mlx.core as mx
import numpy as np

from mlx_video.models.ltx.ltx import LTXModel, X0Model
from mlx_video.models.ltx.transformer import Modality
from mlx_video.models.ltx.video_vae import VideoEncoder, VideoDecoder
from mlx_video.models.ltx.text_encoder import LTX2TextEncoder, load_text_encoder


@dataclass
class GenerationConfig:
    """Configuration for video generation."""
    # Video dimensions
    height: int = 512
    width: int = 512
    num_frames: int = 33  # Must be 1 + 8*k

    # Diffusion parameters
    num_inference_steps: int = 8  # For distilled model (ignored if use_distilled=True)
    guidance_scale: float = 3.0
    use_distilled: bool = True  # Use hardcoded sigma values for distilled model

    # Latent dimensions (computed from video dimensions)
    @property
    def latent_height(self) -> int:
        return self.height // 32

    @property
    def latent_width(self) -> int:
        return self.width // 32

    @property
    def latent_frames(self) -> int:
        return 1 + (self.num_frames - 1) // 8


# Hardcoded sigma values for distilled model (from LTX-2 pipeline)
# These were tuned to match the distillation process
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]

# Scheduler constants for dynamic sigma computation (non-distilled models)
BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


def get_sigmas(
    num_steps: int,
    num_tokens: int,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
    use_distilled: bool = True,
) -> mx.array:
    """Get sigma schedule for diffusion.

    Args:
        num_steps: Number of diffusion steps
        num_tokens: Number of latent tokens (T * H * W)
        max_shift: Maximum shift for sigma schedule
        base_shift: Base shift for sigma schedule
        stretch: Whether to stretch sigmas to terminal value
        terminal: Terminal value for stretching
        use_distilled: If True, use hardcoded distilled sigma values

    Returns:
        Array of sigma values
    """
    import math

    # For distilled model, use hardcoded sigma values
    if use_distilled:
        return mx.array(DISTILLED_SIGMA_VALUES, dtype=mx.float32)

    # For non-distilled models, compute dynamically using LTX2Scheduler logic
    # Linear base schedule
    sigmas = mx.linspace(1.0, 0.0, num_steps + 1)

    # Compute token-dependent sigma shift
    x1 = BASE_SHIFT_ANCHOR
    x2 = MAX_SHIFT_ANCHOR
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    sigma_shift = num_tokens * mm + b

    # Apply exponential transformation
    # sigmas = exp(sigma_shift) / (exp(sigma_shift) + (1/sigmas - 1)^1)
    power = 1
    exp_shift = math.exp(sigma_shift)

    # Convert to numpy for computation then back to mx
    sigmas_np = np.array(sigmas)
    result = np.zeros_like(sigmas_np)
    non_zero = sigmas_np != 0
    result[non_zero] = exp_shift / (exp_shift + (1.0 / sigmas_np[non_zero] - 1.0) ** power)

    # Stretch sigmas so final value matches terminal
    if stretch:
        non_zero_mask = result != 0
        non_zero_sigmas = result[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        result[non_zero_mask] = stretched

    return mx.array(result, dtype=mx.float32)


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
    # These are the starting coordinates for each patch in latent space
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
        # Shift and clamp to keep first-frame timestamps non-negative
        pixel_coords[:, 0, :, :] = np.clip(
            pixel_coords[:, 0, :, :] + 1 - temporal_scale,
            a_min=0,
            a_max=None
        )

    # Convert temporal to time in seconds by dividing by fps
    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps

    return mx.array(pixel_coords, dtype=mx.float32)


class LTXVideoPipeline:

    def __init__(
        self,
        transformer: LTXModel,
        text_encoder: Optional[LTX2TextEncoder] = None,
        tokenizer: Optional[any] = None,
        vae_encoder: Optional[VideoEncoder] = None,
        vae_decoder: Optional[VideoDecoder] = None,
    ):
        """Initialize pipeline.

        Args:
            transformer: LTX transformer model
            text_encoder: Optional LTX text encoder
            tokenizer: Optional tokenizer for text encoding
            vae_encoder: Optional VAE encoder
            vae_decoder: Optional VAE decoder
        """
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.x0_model = X0Model(transformer)

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: mx.Dtype = mx.float16,
    ) -> mx.array:
        """Prepare initial noise latents.

        Args:
            batch_size: Batch size
            num_frames: Number of latent frames
            height: Latent height
            width: Latent width
            dtype: Data type

        Returns:
            Random latent noise
        """
        # Use in_channels from transformer config
        in_channels = self.transformer.config.in_channels
        shape = (batch_size, in_channels, num_frames, height, width)
        latents = mx.random.normal(shape).astype(dtype)
        return latents

    def prepare_text_embeddings(
        self,
        prompt: Union[str, List[str]],
        batch_size: int,
        max_length: int = 1024,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Prepare text embeddings.

        Args:
            prompt: Text prompt or list of prompts
            batch_size: Batch size
            max_length: Maximum sequence length for tokenization

        Returns:
            Tuple of (text_embeddings, attention_mask)
        """
        # If text encoder is available, use it
        if self.text_encoder is not None and self.tokenizer is not None:
            # Handle single or multiple prompts
            if isinstance(prompt, str):
                prompts = [prompt] * batch_size
            else:
                prompts = prompt

            # Tokenize
            tokens = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )

            input_ids = mx.array(tokens["input_ids"])
            attention_mask = mx.array(tokens["attention_mask"])

            # Encode
            embeddings = self.text_encoder(input_ids, attention_mask)
            mx.eval(embeddings)

            return embeddings, None  # Connector handles masking internally

        # Fallback: random embeddings (for testing without text encoder)
        print("Warning: No text encoder provided, using random embeddings")
        seq_len = max_length + 128  # Account for learnable registers
        embed_dim = self.transformer.config.caption_channels

        embeddings = mx.random.normal((batch_size, seq_len, embed_dim))
        mask = mx.ones((batch_size, seq_len))

        return embeddings, mask

    def denoise_step(
        self,
        latents: mx.array,
        sigma: float,
        sigma_next: float,
        text_embeddings: mx.array,
        positions: mx.array,
        text_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Perform one denoising step.

        Args:
            latents: Current noisy latents
            sigma: Current noise level
            sigma_next: Next noise level
            text_embeddings: Text conditioning
            positions: Position grid for RoPE
            text_mask: Optional attention mask for text

        Returns:
            Denoised latents
        """
        batch_size = latents.shape[0]

        # Flatten latents for transformer: (B, C, F, H, W) -> (B, F*H*W, C)
        b, c, f, h, w = latents.shape
        latents_flat = mx.reshape(latents, (b, c, -1))
        latents_flat = mx.transpose(latents_flat, (0, 2, 1))

        # Create timestep tensor
        timesteps = mx.full((batch_size,), sigma)

        # Create video modality input
        video_modality = Modality(
            latent=latents_flat,
            timesteps=timesteps,
            positions=positions,
            context=text_embeddings,
            context_mask=text_mask,
            enabled=True,
        )

        # Run denoising
        denoised_video, _ = self.x0_model(video=video_modality, audio=None)

        # Reshape back: (B, F*H*W, C) -> (B, C, F, H, W)
        denoised_video = mx.transpose(denoised_video, (0, 2, 1))
        denoised_video = mx.reshape(denoised_video, (b, c, f, h, w))

        # Euler step
        if sigma_next > 0:
            # x_next = x0 + sigma_next * (x - x0) / sigma
            noise = (latents - denoised_video) / sigma
            latents = denoised_video + sigma_next * noise
        else:
            latents = denoised_video

        return latents

    def __call__(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        seed: Optional[int] = None,
    ) -> mx.array:
        """Generate video from text prompt.

        Args:
            prompt: Text prompt
            config: Generation configuration
            seed: Random seed

        Returns:
            Generated video tensor of shape (B, C, F, H, W)
        """
        if config is None:
            config = GenerationConfig()

        if seed is not None:
            mx.random.seed(seed)

        batch_size = 1

        # Prepare text embeddings
        text_embeddings, text_mask = self.prepare_text_embeddings(prompt, batch_size)

        # Prepare initial latents
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_frames=config.latent_frames,
            height=config.latent_height,
            width=config.latent_width,
        )

        # Prepare position grid
        positions = create_position_grid(
            batch_size=batch_size,
            num_frames=config.latent_frames,
            height=config.latent_height,
            width=config.latent_width,
        )

        # Get sigma schedule
        num_tokens = config.latent_frames * config.latent_height * config.latent_width
        sigmas = get_sigmas(
            config.num_inference_steps,
            num_tokens,
            use_distilled=config.use_distilled,
        )

        # Denoising loop
        for i in range(len(sigmas) - 1):
            sigma = float(sigmas[i])
            sigma_next = float(sigmas[i + 1])

            latents = self.denoise_step(
                latents=latents,
                sigma=sigma,
                sigma_next=sigma_next,
                text_embeddings=text_embeddings,
                positions=positions,
                text_mask=text_mask,
            )

            mx.eval(latents)

        # Decode latents to video
        if self.vae_decoder is not None:
            video = self.vae_decoder(latents)
        else:
            video = latents

        return video


def generate_video(
    prompt: str,
    transformer: LTXModel,
    text_encoder: Optional[LTX2TextEncoder] = None,
    tokenizer: Optional[any] = None,
    vae_decoder: Optional[VideoDecoder] = None,
    config: Optional[GenerationConfig] = None,
    seed: Optional[int] = None,
) -> mx.array:
    """Generate video from text prompt.

    Args:
        prompt: Text prompt
        transformer: LTX transformer model
        text_encoder: Optional text encoder
        tokenizer: Optional tokenizer
        vae_decoder: Optional VAE decoder
        config: Generation configuration
        seed: Random seed

    Returns:
        Generated video tensor
    """
    pipeline = LTXVideoPipeline(
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae_decoder=vae_decoder,
    )

    return pipeline(prompt, config, seed)


def load_pipeline(
    model_path: str,
    text_encoder_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    load_text_encoder_weights: bool = True,
) -> LTXVideoPipeline:
    """Load complete LTX-2 video generation pipeline.

    Args:
        model_path: Path to LTX-2 model weights (safetensors)
        text_encoder_path: Path to text encoder weights directory
        tokenizer_path: Path to tokenizer directory
        load_text_encoder_weights: Whether to load text encoder weights

    Returns:
        Configured LTXVideoPipeline
    """
    from transformers import AutoTokenizer

    from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType
    from mlx_video.models.ltx.ltx import LTXModel
    from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder
    from mlx_video.convert import sanitize_transformer_weights

    print("Loading LTX-2 pipeline...")

    # Load transformer
    print("  Loading transformer...")
    raw_weights = mx.load(model_path)
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
    )
    transformer = LTXModel(config)
    transformer.load_weights(list(sanitized.items()), strict=False)
    print("  Transformer loaded")

    # Load VAE decoder
    print("  Loading VAE decoder...")
    vae_decoder = load_vae_decoder(model_path, timestep_conditioning=True)
    print("  VAE decoder loaded")

    # Load text encoder if paths provided
    text_encoder = None
    tokenizer = None

    if load_text_encoder_weights and text_encoder_path is not None:
        print("  Loading text encoder...")
        text_encoder = load_text_encoder(model_path, text_encoder_path)
        print("  Text encoder loaded")

    if tokenizer_path is not None:
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("  Tokenizer loaded")

    print("Pipeline ready!")

    return LTXVideoPipeline(
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae_decoder=vae_decoder,
    )


def video_to_numpy(video: mx.array) -> np.ndarray:
    """Convert video tensor to numpy array.

    Args:
        video: Video tensor of shape (B, C, F, H, W) in range [-1, 1]

    Returns:
        Numpy array of shape (B, F, H, W, C) in range [0, 255]
    """
    # Clamp to [-1, 1]
    video = mx.clip(video, -1.0, 1.0)

    # Scale to [0, 255]
    video = ((video + 1.0) / 2.0 * 255.0).astype(mx.uint8)

    # Rearrange: (B, C, F, H, W) -> (B, F, H, W, C)
    video = mx.transpose(video, (0, 2, 3, 4, 1))

    return np.array(video)


if __name__ == "__main__":
    # Example usage
    from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType

    # Create a small test config
    config = LTXModelConfig(
        model_type=LTXModelType.VideoOnly,
        num_layers=2,  # Reduced for testing
        num_attention_heads=4,
        attention_head_dim=32,
    )

    # Create model
    model = LTXModel(config)

    # Generate video
    gen_config = GenerationConfig(
        height=256,
        width=256,
        num_frames=9,
        num_inference_steps=4,
    )

    print("Testing generation pipeline...")
    pipeline = LTXVideoPipeline(transformer=model)

    # This would require proper text embeddings in practice
    # video = pipeline("A cat walking", gen_config, seed=42)
    # print(f"Generated video shape: {video.shape}")

    print("Pipeline initialized successfully!")
