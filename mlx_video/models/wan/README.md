
## Wan2.1 / Wan2.2

Both [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [Wan2.2](https://github.com/Wan-Video/Wan2.2) are text-to-video diffusion models built on a DiT (Diffusion Transformer) backbone with a T5 text encoder and 3D VAE. 

They share the same model architecture — the difference is in the inference pipeline:

| | Wan2.1 | Wan2.2 T2V-14B | Wan2.2 I2V-14B | Wan2.2 TI2V-5B |
|---|--------|--------|--------|--------|
| **Task** | Text-to-Video | Text-to-Video | Image-to-Video | Text+Image-to-Video |
| **Pipeline** | Single model | Dual model | Dual model | Single model |
| **Sizes** | 1.3B, 14B | 14B | 14B | 5B |
| **Steps** | 50 | 40 | 40 | 40 |
| **Guidance** | 5.0 (fixed) | 3.0 / 4.0 | 3.5 / 3.5 | 5.0 (fixed) |
| **Shift** | 5.0 | 12.0 | 5.0 | 5.0 |
| **VAE** | Wan2.1 (z=16) | Wan2.1 (z=16) | Wan2.1 (z=16) + encoder | Wan2.2 (z=48) |

### Step 1: Download Weights

Download the original PyTorch checkpoints:

**Wan2.1 (14B)**
```bash
# From https://github.com/Wan-Video/Wan2.1 or HuggingFace
# Expected directory structure:
# wan21_checkpoint/
#   ├── models_t5_umt5-xxl-enc-bf16.pth
#   ├── Wan2.1_VAE.pth
#   └── diffusion_pytorch_model*.safetensors   # single model
```

**Wan2.1 (1.3B)** — same structure, smaller transformer weights.

**Wan2.2 (14B)**
```bash
# From https://github.com/Wan-Video/Wan2.2 or HuggingFace
# Expected directory structure:
# wan22_checkpoint/
#   ├── models_t5_umt5-xxl-enc-bf16.pth
#   ├── Wan2.1_VAE.pth
#   ├── low_noise_model/   # safetensors
#   └── high_noise_model/  # safetensors
```

**Wan2.2 I2V-14B** — same directory structure as Wan2.2 T2V. The conversion script auto-detects I2V-14B from the model's `config.json` (`model_type: "i2v"`, `in_dim: 36`).

### Step 2: Convert to MLX Format

The conversion script auto-detects the model version based on the directory structure (presence of `low_noise_model/` subdirectory) and model type (`model_type` in source config.json for I2V vs T2V).

```bash
# Auto-detect version
python -m mlx_video.convert_wan \
    --checkpoint-dir /path/to/wan_checkpoint \
    --output-dir wan_mlx

# Explicit version
python -m mlx_video.convert_wan \
    --checkpoint-dir /path/to/wan21_checkpoint \
    --output-dir wan21_mlx \
    --model-version 2.1

python -m mlx_video.convert_wan \
    --checkpoint-dir /path/to/wan22_checkpoint \
    --output-dir wan22_mlx \
    --model-version 2.2
```

#### Conversion Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint-dir` | (required) | Path to original PyTorch checkpoint directory |
| `--output-dir` | `wan_mlx_model` | Output path for MLX model |
| `--dtype` | `bfloat16` | Target dtype (`float16`, `float32`, `bfloat16`) |
| `--model-version` | `auto` | Model version: `2.1`, `2.2`, or `auto` |
| `--quantize` | off | Quantize transformer weights for reduced memory |
| `--bits` | `4` | Quantization bits: `4` or `8` |
| `--group-size` | `64` | Quantization group size: `32`, `64`, or `128` |

The converter produces:
```
wan_mlx/
├── config.json                    # Model configuration
├── t5_encoder.safetensors         # T5 UMT5-XXL text encoder
├── vae.safetensors                # 3D VAE decoder
├── vae_encoder.safetensors        # 3D VAE encoder (I2V-14B only)
├── model.safetensors              # (Wan2.1) Single transformer
├── low_noise_model.safetensors    # (Wan2.2) Low-noise transformer
└── high_noise_model.safetensors   # (Wan2.2) High-noise transformer
```

### Quantization (Reduced Memory)

Quantize the transformer weights to reduce memory usage by ~3.4x. This is especially useful for the 14B model or memory-constrained devices:

```bash
# Convert with 4-bit quantization
python -m mlx_video.convert_wan \
    --checkpoint-dir /path/to/Wan2.1-T2V-1.3B \
    --output-dir wan21_mlx_q4 \
    --quantize --bits 4 --group-size 64

# Generate with quantized model (auto-detected from config.json)
python -m mlx_video.generate_wan \
    --model-dir wan21_mlx_q4 \
    --prompt "A cat playing piano"
```

**What gets quantized**: Self-attention (Q/K/V/O), cross-attention (Q/K/V/O), and FFN (fc1/fc2) — 10 layers × N blocks = ~95% of model weights. Embeddings, norms, and the output head remain in bfloat16 for precision.

| Model | BF16 Size | 4-bit Size | Notes |
|-------|-----------|------------|-------|
| 1.3B | 2.7 GB | 799 MB | ~3.4x smaller |
| 14B | ~28 GB | ~8 GB | Enables running on 16GB devices |

> **Note**: On Apple Silicon, the 1.3B model fits comfortably in unified memory at bf16. Quantization reduces memory but may not speed up inference for small models. For the 14B model, quantization is essential to fit in memory and will also improve speed.

### Wan Model Specifications

**Transformer (14B)**
- 40 layers, 40 attention heads, dim 5120, head dim 128
- 3-way factorized RoPE (temporal + spatial)
- 14.29B parameters

**Transformer (1.3B, Wan2.1 only)**
- 30 layers, 12 attention heads, dim 1536, head dim 128
- Same architecture, smaller scale

**Text Encoder** — UMT5-XXL (5.68B parameters)
- 24 layers, 64 heads, dim 4096, vocab 256K

**VAE** — 3D causal convolution decoder (72.6M parameters)
- Latent channels: 16
- Compression: 4× temporal, 8× spatial

---

## LoRA Support

LoRA's can be used with the `--lora-high` and `--lora-low` command line switches.

For example, for using the the distilled [Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning) LoRA, use the following command. Lightning speeds up generation by using only 4 steps and a CFG scale of 1.

```bash
python -m mlx_video.generate_wan \
    --model-dir /Volumes/SSD/Wan-AI/Wan2.2-T2V-A14B-MLX \
    --width 480 \
    --height 480 \
    --num-frames 121 \
    --prompt "Two dogs of the poodle breed sitting on a beach wearing sunglasses, close up, cinematic, sunset" \
    --steps 4 \
    --guide-scale 1 \
    --trim-first-frames 1 \
    --lora-high /Volumes/SSD/Wan-AI/lightx2v/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/high_noise_model.safetensors 1 \
    --lora-low /Volumes/SSD/Wan-AI/lightx2v/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/low_noise_model.safetensors 1
 ```
