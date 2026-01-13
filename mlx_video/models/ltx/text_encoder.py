"""Gemma 3 Text Encoder for LTX-2 - Full Pipeline.

Uses mlx-vlm's Gemma3 implementation which has been validated to match PyTorch
with 0.999 correlation.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_video.utils import rms_norm, apply_quantization
from mlx_video.models.ltx.rope import apply_interleaved_rotary_emb

from mlx_vlm.models.gemma3.language import Gemma3Model
from mlx_vlm.models.gemma3.config import TextConfig


class LanguageModel(nn.Module):


    def __init__(self, config: TextConfig):
        super().__init__()
        # Create config matching LTX-2 text encoder requirements
        self.config = config 

        # Create the Gemma3Model from mlx-vlm
        self.model = Gemma3Model(self.config)

    def _create_causal_mask_with_padding(
        self,
        seq_len: int,
        attention_mask: Optional[mx.array],
        dtype: mx.Dtype,
    ) -> mx.array:
        
        causal_mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))

        if attention_mask is not None:
            batch_size = attention_mask.shape[0]

            padding_mask = attention_mask.astype(mx.bool_)  # (batch, seq_len)
            combined = causal_mask[None, :, :] & padding_mask[:, None, :]
            min_val = mx.finfo(dtype).min if dtype in (mx.float16, mx.bfloat16) else -1e9
            mask = mx.where(combined, mx.zeros(combined.shape, dtype=dtype),
                           mx.full(combined.shape, min_val, dtype=dtype))
            return mask[:, None, :, :]
        else:
            # No padding mask, just causal
            min_val = mx.finfo(dtype).min if dtype in (mx.float16, mx.bfloat16) else -1e9
            mask = mx.where(causal_mask, mx.zeros((seq_len, seq_len), dtype=dtype),
                           mx.full((seq_len, seq_len), min_val, dtype=dtype))
            return mask[None, None, :, :]  # (1, 1, seq, seq)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: bool = True,
    ) -> Tuple[mx.array, List[mx.array]]:
        """Forward pass returning hidden states.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            attention_mask: Optional attention mask (1 for valid, 0 for padding)
            output_hidden_states: Whether to return all hidden states

        Returns:
            Tuple of (final_hidden_states, list_of_all_hidden_states)
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        h = self.model.embed_tokens(input_ids)

        # Apply Gemma scaling
        h *= mx.array(self.config.hidden_size**0.5, mx.bfloat16).astype(h.dtype)
        mx.eval(h)

        all_hidden_states = [h] if output_hidden_states else []

        # Set up cache (all None for non-cached inference)
        cache = [None] * len(self.model.layers)

        full_causal_mask = self._create_causal_mask_with_padding(seq_len, attention_mask, h.dtype)


        sliding_mask = full_causal_mask


        num_layers = len(self.model.layers)
        for i, layer in enumerate(self.model.layers):
            is_global = (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            )

            # Select appropriate mask for this layer
            if is_global:
                local_mask = full_causal_mask
            else:
                local_mask = sliding_mask

            h = layer(h, local_mask, None)
            mx.eval(h)

            if output_hidden_states and i < num_layers - 1:
                all_hidden_states.append(h)

        # Apply final norm
        hidden_states = self.model.norm(h)
        mx.eval(hidden_states)

        # Append the final normalized output as the last hidden state
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        prefix = "language_model."
        sanitized = {}
        for key, value in weights.items():
            if key.startswith(prefix):
                # If the weight is float32, cast to bfloat16
                if hasattr(value, "dtype") and value.dtype == mx.float32:
                    sanitized[key[len(prefix):]] = value.astype(mx.bfloat16)
                else:
                    sanitized[key[len(prefix):]] = value
        return sanitized
    @classmethod
    def from_pretrained(cls, model_path: str):
        import json
        weight_files = sorted(Path(model_path).glob("*.safetensors"))
        config_file = Path(model_path) / "config.json"
        config_dict = {}
        if config_file.exists():
            with open(config_file, "r") as f:
                config_dict = json.load(f)

            language_model = cls(config=TextConfig.from_dict(config_dict["text_config"]))
        else:
            raise ValueError(f"Config file not found at {model_path}")

        quantization = config_dict.get("quantization", None)
        weights = {}
        for i, wf in enumerate(weight_files):
            weights.update(mx.load(str(wf)))

                
        if hasattr(language_model, "sanitize"):
            weights = language_model.sanitize(weights=weights)


        apply_quantization(model=language_model, weights=weights, quantization=quantization)

        language_model.load_weights(list(weights.items()), strict=False)

        return language_model



class ConnectorAttention(nn.Module):

    def __init__(
        self,
        dim: int = 3840,
        num_heads: int = 30,
        head_dim: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = [nn.Linear(inner_dim, dim, bias=True)]

        # Standard RMSNorm (not Gemma-style) on full inner_dim
        self.q_norm = nn.RMSNorm(inner_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(inner_dim, eps=1e-6)

    def __call__(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
        pe: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.to_q(x)  # (B, seq, inner_dim)
        k = self.to_k(x)
        v = self.to_v(x)

        # QK normalization on full inner_dim BEFORE reshape (matches PyTorch)
        q = self.q_norm(q)
        k = self.k_norm(k)


        if pe is not None:
            # pe: tuple of (cos, sin) each with shape (1, seq_len, inner_dim)
            q = apply_interleaved_rotary_emb(q, pe[0], pe[1])
            k = apply_interleaved_rotary_emb(k, pe[0], pe[1])

        q = mx.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = mx.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = mx.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)

        # No mask needed for connector - after register replacement, all positions are valid
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        return self.to_out[0](out)
    

class GEGLU(nn.Module):
    """GELU-gated linear unit."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu_approx(self.proj(x))


class ConnectorFeedForward(nn.Module):

    def __init__(self, dim: int = 3840, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * mult
        self.net = [
            GEGLU(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=True),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.net:
            x = layer(x)
        return x


class ConnectorTransformerBlock(nn.Module):

    def __init__(self, dim: int = 3840, num_heads: int = 30, head_dim: int = 128):
        super().__init__()
        self.attn1 = ConnectorAttention(dim, num_heads, head_dim)
        self.ff = ConnectorFeedForward(dim)

    def __call__(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
        pe: Optional[mx.array] = None,
    ) -> mx.array:
        # Pre-norm + attention + residual
        norm_x = rms_norm(x)
        if norm_x.ndim == 4:
            norm_x = mx.squeeze(norm_x, axis=1)
        attn_out = self.attn1(norm_x, attention_mask, pe)
        x = x + attn_out
        if x.ndim == 4:
            x = mx.squeeze(x, axis=1)

        # Pre-norm + FFN + residual
        norm_x = rms_norm(x)
        ff_out = self.ff(norm_x)
        x = x + ff_out
        if x.ndim == 4:
            x = mx.squeeze(x, axis=1)

        return x


class Embeddings1DConnector(nn.Module):

    def __init__(
        self,
        dim: int = 3840,
        num_heads: int = 30,
        head_dim: int = 128,
        num_layers: int = 2,
        num_learnable_registers: int = 128,
        positional_embedding_theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_learnable_registers = num_learnable_registers
        self.positional_embedding_theta = positional_embedding_theta

        self.transformer_1d_blocks = [
            ConnectorTransformerBlock(dim, num_heads, head_dim)
            for _ in range(num_layers)
        ]

        if num_learnable_registers > 0:
            self.learnable_registers = mx.zeros((num_learnable_registers, dim))

    def _precompute_freqs_cis(self, seq_len: int, dtype: mx.Dtype) -> Tuple[mx.array, mx.array]:
        """Compute RoPE frequencies for connector (INTERLEAVED type).

        Matches PyTorch: generate_freq_grid_pytorch + generate_freqs + interleaved_freqs_cis
        Returns tuple of (cos, sin) each with shape (1, seq_len, inner_dim).
        """
        import math

        dim = self.num_heads * self.head_dim  # inner_dim = 3840
        theta = self.positional_embedding_theta
        max_pos = [1]  # Default for connector
        n_elem = 2 * len(max_pos)  # = 2

        # Generate frequency indices (matches generate_freq_grid_pytorch)
        start = 1.0
        end = theta
        num_indices = dim // n_elem  # 1920

        log_start = math.log(start) / math.log(theta)  # = 0
        log_end = math.log(end) / math.log(theta)  # = 1
        lin_space = mx.linspace(log_start, log_end, num_indices)
        indices = (theta ** lin_space) * (math.pi / 2)

        # Generate positions and compute freqs (matches generate_freqs)
        positions = mx.arange(seq_len).astype(mx.float32)
        # fractional_positions = positions / max_pos[0] = positions (since max_pos[0]=1)
        # scaled_positions = fractional_positions * 2 - 1 = positions * 2 - 1
        scaled_positions = positions * 2 - 1  # Shape: (seq_len,)

        # freqs = indices * scaled_positions (outer product)
        # Shape: (seq_len, num_indices)
        freqs = scaled_positions[:, None] * indices[None, :]

        # Compute cos/sin with interleaved pattern (matches interleaved_freqs_cis)
        cos_freq = mx.cos(freqs)
        sin_freq = mx.sin(freqs)

        # repeat_interleave: (seq_len, num_indices) -> (seq_len, dim)
        # Pattern: [c0, c0, c1, c1, c2, c2, ...]
        cos_full = mx.repeat(cos_freq, 2, axis=-1)
        sin_full = mx.repeat(sin_freq, 2, axis=-1)

        # Add batch dimension: (1, seq_len, dim)
        cos_full = cos_full[None, :, :]
        sin_full = sin_full[None, :, :]

        return cos_full.astype(dtype), sin_full.astype(dtype)

    def _replace_padded_with_registers(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        batch_size, seq_len, dim = hidden_states.shape

        # Binary mask: 1 for valid tokens, 0 for padded
        # attention_mask is additive: 0 for valid, large negative for padded
        mask_binary = (attention_mask.squeeze(1).squeeze(1) >= -9000.0).astype(mx.int32)  # (batch, seq)

        # Tile registers to match sequence length
        num_tiles = seq_len // self.num_learnable_registers
        registers = mx.tile(self.learnable_registers, (num_tiles, 1))  # (seq_len, dim)

        # Process each batch item (PyTorch uses advanced indexing)
        result_list = []
        for b in range(batch_size):
            mask_b = mask_binary[b]  # (seq,)
            hs_b = hidden_states[b]  # (seq, dim)

            # Count valid tokens
            num_valid = int(mx.sum(mask_b))

            # Extract valid tokens (where mask is 1)
            # Since we have left-padded input, valid tokens are at the end
            valid_tokens = hs_b[seq_len - num_valid:]  # (num_valid, dim)

            # Pad with zeros on the right to get back to seq_len
            pad_length = seq_len - num_valid
            if pad_length > 0:
                padding = mx.zeros((pad_length, dim), dtype=hs_b.dtype)
                adjusted = mx.concatenate([valid_tokens, padding], axis=0)  # (seq_len, dim)
            else:
                adjusted = valid_tokens

            # Create flipped mask: 1s at front (where valid tokens now are), 0s at back
            flipped_mask = mx.concatenate([
                mx.ones((num_valid,), dtype=mx.int32),
                mx.zeros((pad_length,), dtype=mx.int32)
            ], axis=0)  # (seq,)

            # Combine: valid tokens at front, registers at back
            flipped_mask_expanded = flipped_mask[:, None].astype(hs_b.dtype)  # (seq, 1)
            combined = flipped_mask_expanded * adjusted + (1 - flipped_mask_expanded) * registers

            result_list.append(combined)

        hidden_states = mx.stack(result_list, axis=0)  # (batch, seq, dim)

        # Reset attention mask to all zeros (no masking after register replacement)
        attention_mask = mx.zeros_like(attention_mask)

        return hidden_states, attention_mask

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        
        # Replace padded tokens with learnable registers
        if self.num_learnable_registers > 0 and attention_mask is not None:
            hidden_states, attention_mask = self._replace_padded_with_registers(
                hidden_states, attention_mask
            )

        # Compute RoPE frequencies
        seq_len = hidden_states.shape[1]
        freqs_cis = self._precompute_freqs_cis(seq_len, hidden_states.dtype)

        # Process through transformer blocks
        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, attention_mask, freqs_cis)

        # Final RMS norm
        hidden_states = rms_norm(hidden_states)

        return hidden_states, attention_mask



def norm_and_concat_hidden_states(
    hidden_states: List[mx.array],
    attention_mask: mx.array,
    padding_side: str = "left",
) -> mx.array:

    # Stack hidden states: (batch, seq, dim, num_layers)
    stacked = mx.stack(hidden_states, axis=-1)
    b, t, d, num_layers = stacked.shape

    # Compute sequence lengths from attention mask
    sequence_lengths = mx.sum(attention_mask, axis=-1)  # (batch,)

    # Build mask based on padding side
    token_indices = mx.arange(t)[None, :]  # (1, T)

    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]  # (B, T)
    else:  # left padding
        start_indices = t - sequence_lengths[:, None]  # (B, 1)
        mask = token_indices >= start_indices  # (B, T)

    mask = mask[:, :, None, None]  # (B, T, 1, 1)
    eps = 1e-6

    # Compute masked mean per layer
    masked = mx.where(mask, stacked, mx.zeros_like(stacked))
    denom = (sequence_lengths * d).reshape(b, 1, 1, 1)
    mean = mx.sum(masked, axis=(1, 2), keepdims=True) / (denom + eps)

    # Compute masked min/max per layer
    x_for_min = mx.where(mask, stacked, mx.full(stacked.shape, float('inf'), dtype=stacked.dtype))
    x_for_max = mx.where(mask, stacked, mx.full(stacked.shape, float('-inf'), dtype=stacked.dtype))
    x_min = mx.min(x_for_min, axis=(1, 2), keepdims=True)
    x_max = mx.max(x_for_max, axis=(1, 2), keepdims=True)
    range_val = x_max - x_min

    # Normalize: 8 * (x - mean) / range
    normed = 8 * (stacked - mean) / (range_val + eps)

    # Flatten layers into feature dimension: (B, T, D*L)
    normed = mx.reshape(normed, (b, t, -1))

    # Zero out padded positions
    mask_flat = mx.broadcast_to(mask[:, :, :, 0], (b, t, d * num_layers))
    normed = mx.where(mask_flat, normed, mx.zeros_like(normed))

    return normed


class GemmaFeaturesExtractor(nn.Module):

    def __init__(self, input_dim: int = 188160, output_dim: int = 3840):
        super().__init__()
        self.aggregate_embed = nn.Linear(input_dim, output_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.aggregate_embed(x)





class LTX2TextEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 3840,
        num_layers: int = 49,  # 48 transformer layers + 1 embedding
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.language_model = None

        # Feature extractor: 3840*49 -> 3840
        self.feature_extractor = GemmaFeaturesExtractor(
            input_dim=hidden_dim * num_layers,
            output_dim=hidden_dim,
        )

        # Video embeddings connector: 2-layer transformer
        self.video_embeddings_connector = Embeddings1DConnector(
            dim=hidden_dim,
            num_heads=30,
            head_dim=128,
            num_layers=2,
            num_learnable_registers=128,
        )

        self.processor = None

    def load(self, model_path: Optional[str] = None, text_encoder_path: Optional[str] = "google/gemma-3-12b-it"):

        if Path(text_encoder_path / "text_encoder").is_dir():
            text_encoder_path = str(text_encoder_path / "text_encoder")
        
        self.language_model = LanguageModel.from_pretrained(text_encoder_path)
        print(f"Language model data type: {self.language_model.model.embed_tokens.weight.dtype}")

        # Load transformer weights for feature extractor and connector

        transformer_files = list(model_path.glob("ltx-2-19*.safetensors"))
        if transformer_files:
            transformer_weights = mx.load(str(transformer_files[0]))

            # Load feature extractor (aggregate_embed)
            if "text_embedding_projection.aggregate_embed.weight" in transformer_weights:
                self.feature_extractor.aggregate_embed.weight = transformer_weights[
                    "text_embedding_projection.aggregate_embed.weight"
                ]


            # Load video_embeddings_connector weights
            connector_weights = {}
            for key, value in transformer_weights.items():
                if key.startswith("model.diffusion_model.video_embeddings_connector."):
                    new_key = key.replace("model.diffusion_model.video_embeddings_connector.", "")
                    connector_weights[new_key] = value

            if connector_weights:
                # Map weight names to our structure
                mapped_weights = {}
                for key, value in connector_weights.items():
                    # transformer_1d_blocks.X.attn1.* -> transformer_1d_blocks.X.attn1.*
                    # transformer_1d_blocks.X.ff.net.0.proj.* -> transformer_1d_blocks.X.ff.net.0.proj.*
                    # transformer_1d_blocks.X.ff.net.2.* -> transformer_1d_blocks.X.ff.net.2.*
                    mapped_weights[key] = value

                self.video_embeddings_connector.load_weights(
                    list(mapped_weights.items()), strict=False
                )

                # Manually load learnable_registers (it's a plain mx.array, not a parameter)
                if "learnable_registers" in connector_weights:
                    self.video_embeddings_connector.learnable_registers = connector_weights["learnable_registers"]

        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer_path = model_path / "tokenizer"
        if tokenizer_path.exists():
            self.processor = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
        else:
            self.processor = AutoTokenizer.from_pretrained(text_encoder_path, trust_remote_code=True)
        # Set left padding to match official LTX-2 text encoder
        self.processor.padding_side = "left"

        print("Text encoder loaded successfully")

    def encode(
        self,
        prompt: str,
        max_length: int = 1024,
    ) -> Tuple[mx.array, mx.array]:

        if self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.processor(
            prompt,
            return_tensors="np",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])

        _, all_hidden_states = self.language_model(input_ids, attention_mask, output_hidden_states=True)

        concat_hidden = norm_and_concat_hidden_states(
            all_hidden_states, attention_mask, padding_side="left"
        )

        features = self.feature_extractor(concat_hidden)

        additive_mask = (attention_mask - 1).astype(features.dtype)
        additive_mask = additive_mask.reshape(attention_mask.shape[0], 1, 1, -1) * 1e9

        embeddings, _ = self.video_embeddings_connector(features, additive_mask)

        return embeddings, attention_mask

    def __call__(
        self,
        prompt: str,
        max_length: int = 1024,
    ) -> Tuple[mx.array, mx.array]:
        return self.encode(prompt, max_length)


def load_text_encoder(model_path: str = "/tmp/ltx2") -> LTX2TextEncoder:
    encoder = LTX2TextEncoder(model_path=model_path)
    encoder.load()
    return encoder

