import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from mlx_video.models.ltx import LTXModel, LTXModelConfig  # noqa: E402
from mlx_video.convert import (  # noqa: E402
    load_transformer_weights,
    load_vae_weights,
)
from mlx_video.generate import generate_video  # noqa: E402


def generate_video_with_audio(*args, **kwargs):
    """Lazy import to avoid preloading the CLI target module."""
    from mlx_video.generate_av import generate_video_with_audio as _impl

    return _impl(*args, **kwargs)


__all__ = [
    # Models
    "LTXModel",
    "LTXModelConfig",
    # Weight loaders
    "load_transformer_weights",
    "load_vae_weights",
    # Generation functions
    "generate_video",
    "generate_video_with_audio",
]
