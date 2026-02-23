"""Standalone prompt enhancement using uncensored Gemma via mlx_lm."""

import re
import sys
from pathlib import Path
from typing import Optional

PROMPTS_DIR = Path(__file__).parent / "prompts"
UNCENSORED_MODEL_REPO = "TheCluster/amoral-gemma-3-12B-v2-mlx-4bit"


def _load_system_prompt(prompt_name: str) -> str:
    """Load a system prompt from the prompts directory."""
    prompt_path = PROMPTS_DIR / prompt_name
    if prompt_path.exists():
        return prompt_path.read_text().strip()
    raise FileNotFoundError(f"System prompt not found: {prompt_path}")


def _apply_chat_template(system_prompt: str, user_content: str) -> str:
    """Apply Gemma 3 chat template."""
    formatted = f"<start_of_turn>user\n{system_prompt}<end_of_turn>\n"
    formatted += f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
    formatted += "<start_of_turn>model\n"
    return formatted


def _clean_response(response: str) -> str:
    """Clean up the generated response."""
    response = response.strip()
    response = re.sub(r"^[^\w\s]+", "", response)
    return response


def enhance_with_model(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    seed: int = 42,
    max_tokens: int = 512,
    verbose: bool = True,
) -> str:
    """Enhance a text prompt using uncensored Gemma 12B via mlx_lm.

    Args:
        prompt: The original user prompt
        system_prompt: Optional custom system prompt (default: gemma_t2v)
        temperature: Sampling temperature
        seed: Random seed
        max_tokens: Max tokens to generate
        verbose: Whether to show progress

    Returns:
        Enhanced prompt string
    """
    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        print(
            "mlx-lm not available for uncensored enhancement. Using original prompt.",
            file=sys.stderr,
        )
        return prompt

    print(
        "Loading uncensored prompt enhancer (first run may download ~7GB)...",
        file=sys.stderr,
        flush=True,
    )

    model, tokenizer = load(UNCENSORED_MODEL_REPO)

    system_prompt = system_prompt or _load_system_prompt("gemma_t2v_system_prompt.txt")
    user_content = f"user prompt: {prompt}"
    formatted = _apply_chat_template(system_prompt, user_content)

    import mlx.core as mx

    mx.random.seed(seed)

    # mlx-lm 0.25+ uses sampler instead of temp kwarg (generate_step rejects temp)
    sampler = make_sampler(temperature, 1.0, 0.0, 1, top_k=0)
    response = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=verbose,
    )

    del model
    mx.clear_cache()

    enhanced = _clean_response(response)
    return enhanced
