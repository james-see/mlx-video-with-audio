# mlx-video

MLX-Video is the best package for inference and finetuning of Image-Video-Audio generation models on your Mac using MLX.

## Features

- Text-to-Video (T2V) generation with synchronized audio
- Image-to-Video (I2V) generation with synchronized audio
- Video-only generation (without audio)
- Two-stage generation pipeline for high-quality output
- 2x spatial upscaling for images and videos
- Optimized for Apple Silicon using MLX
- Cross-modal attention for audio-video synchronization

## Installation

### Option 1: Install with pip (requires git):

```bash
pip install git+https://github.com/Blaizzy/mlx-video.git
```

### Option 2: Install with uv (ultra-fast package manager):

```bash
uv pip install git+https://github.com/Blaizzy/mlx-video.git
```

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python >= 3.11
- MLX >= 0.22.0
- ffmpeg (for audio-video muxing)

Install ffmpeg if not already installed:

```bash
brew install ffmpeg
```

## Supported Models

### LTX-2

[LTX-2](https://huggingface.co/Lightricks/LTX-Video) is a 19B parameter video generation model from Lightricks with audio generation capabilities.

> **Note:** Currently, only the distilled variant (`ltx-2-19b-distilled`) is supported.

## Usage

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and isolation.

### Text-to-Video with Audio (T2V+Audio)

Generate videos with synchronized audio from text descriptions:

```bash
uv run mlx_video.generate_av --prompt "A jazz band playing in a smoky club"
```

With custom settings:

```bash
uv run mlx_video.generate_av \
    --prompt "Ocean waves crashing on a beach at sunset" \
    --height 768 \
    --width 768 \
    --num-frames 65 \
    --seed 123 \
    --output-path my_video.mp4
```

### Image-to-Video with Audio (I2V+Audio)

Generate videos from an input image with synchronized audio:

```bash
uv run mlx_video.generate_av \
    --prompt "A person dancing to upbeat music" \
    --image photo.jpg \
    --image-strength 0.8
```

### Video-Only Generation (no audio)

For video generation without audio:

```bash
uv run mlx_video.generate --prompt "Two dogs of the poodle breed wearing sunglasses, close up, cinematic, sunset" -n 100 --width 768
```

<img src="https://github.com/Blaizzy/mlx-video/raw/main/examples/poodles.gif" width="512" alt="Poodles demo">

## CLI Reference

### generate_av (Audio-Video)

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | (required) | Text description of the video/audio |
| `--height`, `-H` | 512 | Output height (must be divisible by 64) |
| `--width`, `-W` | 512 | Output width (must be divisible by 64) |
| `--num-frames`, `-n` | 65 | Number of frames (must be 1 + 8*k) |
| `--seed`, `-s` | 42 | Random seed for reproducibility |
| `--fps` | 24 | Frames per second |
| `--output-path` | output_av.mp4 | Output video path |
| `--output-audio` | (auto) | Output audio path (default: same as video with .wav) |
| `--image`, `-i` | None | Path to conditioning image for I2V |
| `--image-strength` | 1.0 | Conditioning strength (1.0 = full denoise) |
| `--image-frame-idx` | 0 | Frame index to condition (0 = first frame) |
| `--enhance-prompt` | false | Enhance prompt using Gemma |
| `--tiling` | auto | Tiling mode for VAE (auto/none/default/aggressive/conservative) |
| `--model-repo` | Lightricks/LTX-2 | HuggingFace model repository |

### generate (Video-Only)

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | (required) | Text description of the video |
| `--height`, `-H` | 512 | Output height (must be divisible by 64) |
| `--width`, `-W` | 512 | Output width (must be divisible by 64) |
| `--num-frames`, `-n` | 100 | Number of frames |
| `--seed`, `-s` | 42 | Random seed for reproducibility |
| `--fps` | 24 | Frames per second |
| `--output`, `-o` | output.mp4 | Output video path |
| `--save-frames` | false | Save individual frames as images |
| `--model-repo` | Lightricks/LTX-2 | HuggingFace model repository |

### convert (Model Conversion)

Convert HuggingFace models to unified MLX format for faster loading:

```bash
uv run mlx_video.convert --hf-path Lightricks/LTX-2 --mlx-path ~/models/ltx2-mlx-av
```

| Option | Default | Description |
|--------|---------|-------------|
| `--hf-path` | Lightricks/LTX-2 | HuggingFace model path or repo ID |
| `--mlx-path` | mlx_model | Output path for MLX model |
| `--dtype` | bfloat16 | Target dtype (float16/float32/bfloat16) |
| `--no-audio` | false | Exclude audio components (video-only model) |

## Using Pre-Converted MLX Models

For faster loading, you can use pre-converted MLX models instead of converting on-the-fly.

### Option 1: Use a Pre-Converted Model from HuggingFace

```bash
# Use a community-converted MLX model (replace with actual repo)
uv run mlx_video.generate_av \
    --prompt "A jazz band playing in a smoky club" \
    --model-repo username/ltx2-mlx-av
```

### Option 2: Convert Your Own Model

1. **Convert the model** (one-time, ~42GB output):

```bash
uv run mlx_video.convert --hf-path Lightricks/LTX-2 --mlx-path ~/models/ltx2-mlx-av
```

2. **Use the converted model**:

```bash
uv run mlx_video.generate_av \
    --prompt "A jazz band playing in a smoky club" \
    --model-repo ~/models/ltx2-mlx-av
```

### Benefits of Unified MLX Format

- **Faster loading**: Single file vs multiple scattered files
- **Pre-sanitized weights**: No on-the-fly key transformation
- **Smaller footprint**: Only includes necessary weights (no quantized variants)
- **Easy sharing**: Upload to HuggingFace for others to use

## How It Works

### Video Generation Pipeline

The pipeline uses a two-stage generation process:

1. **Stage 1**: Generate at half resolution (e.g., 384x384) with 8 denoising steps
2. **Upsample**: 2x spatial upsampling via LatentUpsampler
3. **Stage 2**: Refine at full resolution (e.g., 768x768) with 3 denoising steps
4. **Decode**: VAE decoder converts latents to RGB video

### Audio Generation Pipeline

Audio is generated in sync with video through:

1. **Joint Denoising**: Video and audio latents are denoised together
2. **Cross-Modal Attention**: Bidirectional attention between video and audio
3. **Audio Decoding**: Audio VAE converts latents to mel spectrogram
4. **Vocoder**: HiFi-GAN converts mel spectrogram to waveform
5. **Muxing**: ffmpeg combines video and audio

### Architecture

```
Text Prompt
    │
    ▼
┌─────────────────────────────────────────────┐
│          Text Encoder (Gemma 3 12B)         │
│  ┌─────────────┐      ┌─────────────┐       │
│  │   Video     │      │   Audio     │       │
│  │ Connector   │      │ Connector   │       │
│  │  (4096-dim) │      │  (2048-dim) │       │
│  └──────┬──────┘      └──────┬──────┘       │
└─────────┼────────────────────┼──────────────┘
          │                    │
          ▼                    ▼
┌─────────────────────────────────────────────┐
│        LTX Transformer (48 layers)          │
│  ┌─────────────┐ ◄──► ┌─────────────┐       │
│  │ Video Path  │      │ Audio Path  │       │
│  │  (4096-dim) │      │  (2048-dim) │       │
│  └──────┬──────┘      └──────┬──────┘       │
└─────────┼────────────────────┼──────────────┘
          │                    │
          ▼                    ▼
┌─────────────────┐    ┌─────────────────┐
│   Video VAE     │    │   Audio VAE     │
│   Decoder       │    │   Decoder       │
└────────┬────────┘    └────────┬────────┘
         │                      │
         ▼                      ▼
    Video Frames          ┌─────────────┐
                          │   Vocoder   │
                          │  (HiFi-GAN) │
                          └──────┬──────┘
                                 │
                                 ▼
                           Audio Waveform
```

## Model Specifications

### Video Path

- **Transformer**: 48 layers, 32 attention heads, 128 dim per head (4096 total)
- **Latent channels**: 128
- **Text encoder**: Gemma 3 with 3840-dim features, projected to 4096-dim
- **RoPE**: Split mode with double precision

### Audio Path

- **Transformer**: 48 layers, 32 attention heads, 64 dim per head (2048 total)
- **Latent channels**: 8 (patchified to 128)
- **Mel bins**: 16 (latent), 64 (decoded)
- **Sample rate**: 24kHz output, 16kHz internal
- **Audio latents per second**: 25

### Cross-Modal Attention

- Bidirectional attention between video and audio paths
- Separate timestep conditioning for cross-attention
- Gated attention output for controlled mixing

## Project Structure

```
mlx_video/
├── generate.py             # Video-only generation pipeline
├── generate_av.py          # Audio-video generation pipeline
├── convert.py              # Weight conversion (PyTorch -> MLX)
├── postprocess.py          # Video post-processing utilities
├── utils.py                # Helper functions
├── conditioning/           # I2V conditioning utilities
└── models/
    └── ltx/
        ├── ltx.py          # Main LTXModel (DiT transformer)
        ├── config.py       # Model configuration
        ├── transformer.py  # Transformer blocks with cross-modal attention
        ├── attention.py    # Multi-head attention with RoPE
        ├── text_encoder.py # Text encoder with video/audio connectors
        ├── upsampler.py    # 2x spatial upsampler
        ├── video_vae/      # Video VAE encoder/decoder
        └── audio_vae/      # Audio VAE decoder and vocoder
```

## Tips for Best Results

1. **Prompt Quality**: Use detailed, descriptive prompts that include both visual and audio elements
2. **Frame Count**: Use frame counts of the form `1 + 8*k` (e.g., 33, 65, 97) for optimal quality
3. **Resolution**: Higher resolutions (768x768) produce better results but require more memory
4. **Tiling**: For large videos, use `--tiling aggressive` to reduce memory usage
5. **Audio Sync**: Audio is automatically synchronized to video duration

## License

MIT
