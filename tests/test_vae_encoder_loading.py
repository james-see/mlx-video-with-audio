"""Tests for VAE encoder weight loading across model formats (issue #47).

Verifies that load_vae_encoder correctly handles:
- Single-file unified (model.safetensors with vae_encoder. prefix)
- Split-shard unified (vae_encoder.safetensors with bare keys)
- Per-channel statistics in both underscore and hyphen key formats
- Conv3d weight transpose is skipped for MLX-layout weights
"""

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import pytest


def _make_encoder_weights(prefix="", mlx_layout=True):
    """Create minimal fake encoder weights for testing load_vae_encoder.

    Returns dict of weight tensors matching VideoEncoder's expected keys.
    """
    weights = {}
    if mlx_layout:
        shape_5d = (128, 3, 3, 3, 128)  # (O, D, H, W, I) — MLX
    else:
        shape_5d = (128, 128, 3, 3, 3)  # (O, I, D, H, W) — PyTorch

    p = prefix
    weights[f"{p}conv_in.conv.weight"] = mx.zeros(
        (128, 3, 3, 3, 48) if mlx_layout else (128, 48, 3, 3, 3)
    )
    weights[f"{p}conv_in.conv.bias"] = mx.zeros((128,))
    weights[f"{p}conv_out.conv.weight"] = mx.zeros(
        (129, 3, 3, 3, 1024) if mlx_layout else (129, 1024, 3, 3, 3)
    )
    weights[f"{p}conv_out.conv.bias"] = mx.zeros((129,))

    for blk in range(4):
        for conv in ["conv1", "conv2"]:
            weights[f"{p}down_blocks.0.res_blocks.{blk}.{conv}.conv.weight"] = mx.zeros(
                shape_5d
            )
            weights[f"{p}down_blocks.0.res_blocks.{blk}.{conv}.conv.bias"] = mx.zeros(
                (128,)
            )
    return weights


def _write_safetensors(path, weights):
    """Write a dict of mx arrays as a safetensors file."""
    mx.save_safetensors(str(path), weights)


def _write_embedded_config(directory):
    """Write a minimal embedded_config.json with encoder_blocks."""
    cfg = {
        "vae": {
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
            "patch_size": 4,
            "spatial_padding_mode": "zeros",
        }
    }
    with open(Path(directory) / "embedded_config.json", "w") as f:
        json.dump(cfg, f)


class TestLoadVaeEncoderSplitShard:
    """Test loading from split vae_encoder.safetensors (bare keys, MLX layout)."""

    def test_bare_keys_loaded(self):
        from mlx_video.models.ltx.video_vae.encoder import load_vae_encoder

        with tempfile.TemporaryDirectory() as tmp:
            weights = _make_encoder_weights(prefix="", mlx_layout=True)
            weights["per_channel_statistics._mean_of_means"] = mx.zeros((128,))
            weights["per_channel_statistics._std_of_means"] = mx.ones((128,))
            _write_safetensors(Path(tmp) / "vae_encoder.safetensors", weights)
            _write_embedded_config(tmp)

            encoder = load_vae_encoder(tmp, use_unified=True)
            assert encoder.per_channel_statistics.mean is not None
            assert encoder.per_channel_statistics.std is not None

    def test_no_transpose_for_mlx_layout(self):
        """Conv3d weights should NOT be transposed when already in MLX layout."""
        from mlx_video.models.ltx.video_vae.encoder import load_vae_encoder

        with tempfile.TemporaryDirectory() as tmp:
            weights = _make_encoder_weights(prefix="", mlx_layout=True)
            original_shape = weights["conv_in.conv.weight"].shape
            weights["per_channel_statistics._mean_of_means"] = mx.zeros((128,))
            weights["per_channel_statistics._std_of_means"] = mx.ones((128,))
            _write_safetensors(Path(tmp) / "vae_encoder.safetensors", weights)
            _write_embedded_config(tmp)

            encoder = load_vae_encoder(tmp, use_unified=True)
            loaded = dict(encoder.conv_in.conv.parameters())
            assert loaded["weight"].shape == original_shape


class TestLoadVaeEncoderUnifiedSingle:
    """Test loading from single model.safetensors with vae_encoder. prefix."""

    def test_prefixed_keys_loaded(self):
        from mlx_video.models.ltx.video_vae.encoder import load_vae_encoder

        with tempfile.TemporaryDirectory() as tmp:
            weights = _make_encoder_weights(prefix="vae_encoder.", mlx_layout=True)
            weights["vae_encoder.per_channel_statistics._mean_of_means"] = mx.zeros(
                (128,)
            )
            weights["vae_encoder.per_channel_statistics._std_of_means"] = mx.ones(
                (128,)
            )
            # Add some non-encoder keys that should be filtered out
            weights["transformer.layers.0.weight"] = mx.zeros((128, 128))
            _write_safetensors(Path(tmp) / "model.safetensors", weights)

            encoder = load_vae_encoder(tmp, use_unified=True)
            assert encoder.per_channel_statistics.mean is not None
            assert encoder.per_channel_statistics.std is not None


class TestStatsKeyFormats:
    """Test that per-channel statistics are found in all key naming conventions."""

    def test_underscore_format(self):
        from mlx_video.models.ltx.video_vae.encoder import load_vae_encoder

        with tempfile.TemporaryDirectory() as tmp:
            weights = _make_encoder_weights(prefix="", mlx_layout=True)
            weights["per_channel_statistics._mean_of_means"] = mx.full((128,), 0.5)
            weights["per_channel_statistics._std_of_means"] = mx.full((128,), 0.1)
            _write_safetensors(Path(tmp) / "vae_encoder.safetensors", weights)
            _write_embedded_config(tmp)

            encoder = load_vae_encoder(tmp, use_unified=True)
            assert float(encoder.per_channel_statistics.mean[0]) == pytest.approx(0.5)
            assert float(encoder.per_channel_statistics.std[0]) == pytest.approx(0.1)

    def test_hyphen_format(self):
        from mlx_video.models.ltx.video_vae.encoder import load_vae_encoder

        with tempfile.TemporaryDirectory() as tmp:
            weights = _make_encoder_weights(prefix="encoder.", mlx_layout=False)
            weights["per_channel_statistics.mean-of-means"] = mx.full((128,), 0.3)
            weights["per_channel_statistics.std-of-means"] = mx.full((128,), 0.2)
            _write_safetensors(Path(tmp) / "ltx-2-19b-distilled.safetensors", weights)

            encoder = load_vae_encoder(tmp, use_unified=False)
            assert float(encoder.per_channel_statistics.mean[0]) == pytest.approx(0.3)
            assert float(encoder.per_channel_statistics.std[0]) == pytest.approx(0.2)
