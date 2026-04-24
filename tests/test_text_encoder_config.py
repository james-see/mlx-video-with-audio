"""Tests for text encoder config validation (issue #45)."""

import json
import tempfile
from pathlib import Path

import pytest


class TestLooksLikeTextConfig:
    """Tests for _looks_like_text_config heuristic."""

    def _fn(self, config_dict):
        from mlx_video.generate_av import _looks_like_text_config

        return _looks_like_text_config(config_dict)

    def test_gemma3_multimodal_config(self):
        """Gemma 3 multimodal wrapper with nested text_config."""
        config = {
            "architectures": ["Gemma3ForConditionalGeneration"],
            "model_type": "gemma3",
            "text_config": {
                "hidden_size": 3840,
                "intermediate_size": 15360,
                "model_type": "gemma3_text",
                "num_attention_heads": 16,
                "num_hidden_layers": 48,
                "num_key_value_heads": 8,
            },
            "vision_config": {},
        }
        assert self._fn(config) is True

    def test_flat_text_encoder_config(self):
        """Flat text encoder config without nested text_config."""
        config = {
            "hidden_size": 3840,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 262144,
        }
        assert self._fn(config) is True

    def test_flat_without_vocab_size(self):
        """Flat config missing vocab_size should still pass."""
        config = {
            "hidden_size": 3840,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        }
        assert self._fn(config) is True

    def test_gemma_model_type_without_text_config(self):
        """A config with model_type=gemma* should pass even without text_config key."""
        config = {
            "model_type": "gemma3_text",
            "hidden_size": 3840,
        }
        assert self._fn(config) is True

    def test_av_model_config_rejected(self):
        """AV model configs should NOT pass."""
        config = {
            "model_type": "AudioVideo",
            "num_attention_heads": 32,
            "attention_head_dim": 128,
            "audio_mel_bins": 16,
        }
        assert self._fn(config) is False

    def test_empty_config_rejected(self):
        assert self._fn({}) is False

    def test_unrelated_config_rejected(self):
        config = {"foo": "bar", "baz": 42}
        assert self._fn(config) is False


class TestExtractTextConfig:
    """Tests for _extract_text_config in text_encoder.py."""

    def _fn(self, config_dict, model_path=None):
        from mlx_video.models.ltx.text_encoder import _extract_text_config

        return _extract_text_config(config_dict, model_path or Path("/tmp/fake"))

    def test_nested_text_config(self):
        inner = {"hidden_size": 3840, "num_hidden_layers": 48}
        result = self._fn({"text_config": inner, "vision_config": {}})
        assert result is inner

    def test_flat_config(self):
        config = {
            "hidden_size": 3840,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        }
        assert self._fn(config) is config

    def test_av_model_raises_specific_error(self):
        config = {
            "model_type": "AudioVideo",
            "audio_mel_bins": 16,
            "num_attention_heads": 32,
        }
        with pytest.raises(ValueError, match="AV model config"):
            self._fn(config)

    def test_unknown_config_raises(self):
        with pytest.raises(ValueError, match="missing expected keys"):
            self._fn({"random_key": 123})


class TestIsAvModelConfig:
    """Tests for _is_av_model_config."""

    def _fn(self, config_dict):
        from mlx_video.generate_av import _is_av_model_config

        return _is_av_model_config(config_dict)

    def test_audio_video_type(self):
        assert self._fn({"model_type": "AudioVideo"}) is True

    def test_audio_mel_bins_key(self):
        assert self._fn({"audio_mel_bins": 16}) is True

    def test_gemma_config(self):
        assert self._fn({"model_type": "gemma3", "text_config": {}}) is False


class TestValidateTextEncoderConfig:
    """Tests for validate_text_encoder_config with on-disk configs."""

    def _write_config(self, tmp_dir, config_dict):
        config_path = Path(tmp_dir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f)

    def test_valid_gemma_config(self):
        from mlx_video.generate_av import validate_text_encoder_config

        with tempfile.TemporaryDirectory() as tmp:
            self._write_config(
                tmp,
                {
                    "model_type": "gemma3",
                    "text_config": {
                        "hidden_size": 3840,
                        "num_hidden_layers": 48,
                    },
                },
            )
            validate_text_encoder_config(Path(tmp))

    def test_missing_config_raises(self):
        from mlx_video.generate_av import validate_text_encoder_config

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="not found"):
                validate_text_encoder_config(Path(tmp))

    def test_av_config_raises_specific_error(self):
        from mlx_video.generate_av import validate_text_encoder_config

        with tempfile.TemporaryDirectory() as tmp:
            self._write_config(
                tmp,
                {"model_type": "AudioVideo", "audio_mel_bins": 16},
            )
            with pytest.raises(ValueError, match="AV model config"):
                validate_text_encoder_config(Path(tmp))
