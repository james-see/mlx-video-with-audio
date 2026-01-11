"""Gemma 3 Text Encoder for LTX-2 - Full Pipeline."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_video.utils import rms_norm
from mlx_video.models.ltx.rope import apply_rotary_emb_1d

@dataclass
class Gemma3Config:
    """Configuration for Gemma 3 text model."""
    hidden_size: int = 3840
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    vocab_size: int = 262208
    max_position_embeddings: int = 131072


class RMSNorm(nn.Module):
    """RMS Normalization (Gemma style with 1+weight scaling)."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Gemma initializes to ones, but uses (1+weight) scaling
        # After loading weights, weight will have the actual learned values
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        # Gemma-style RMSNorm uses (1 + weight) as the scale factor
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


def apply_rotary_emb(
    q: mx.array,
    k: mx.array,
    positions: mx.array,
    head_dim: int,
    rope_theta: float = 1000000.0,
) -> Tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to Q and K."""
    inv_freq = 1.0 / (rope_theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    freqs = positions[:, :, None].astype(mx.float32) * inv_freq[None, None, :]
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    cos_full = mx.concatenate([cos, cos], axis=-1)
    sin_full = mx.concatenate([sin, sin], axis=-1)
    q_embed = q * cos_full + rotate_half(q) * sin_full
    k_embed = k * cos_full + rotate_half(k) * sin_full
    return q_embed, k_embed




class Gemma3MLP(nn.Module):
    """Gemma 3 MLP with gated activation."""

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.gelu_approx(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Gemma3Attention(nn.Module):

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(config.head_dim)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        positions: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = mx.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = mx.reshape(k, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = mx.reshape(v, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = apply_rotary_emb(q, k, positions, self.head_dim, self.config.rope_theta)

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Create causal mask (lower triangular)
        causal_mask = mx.triu(mx.full((seq_len, seq_len), -1e9, dtype=k.dtype), k=1)
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, seq, seq

        if attention_mask is not None:
            causal_mask = causal_mask + (1.0 - attention_mask[:, None, None, :].astype(k.dtype)) * -1e9

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=causal_mask)
        out = mx.transpose(out, (0, 2, 1, 3))
        out = mx.reshape(out, (batch_size, seq_len, -1))

        return self.o_proj(out)


class Gemma3DecoderLayer(nn.Module):

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.self_attn = Gemma3Attention(config)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        positions: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, attention_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3TextModel(nn.Module):

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Gemma scales embeddings by sqrt(hidden_size)
        self.embed_scale = config.hidden_size ** 0.5

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: bool = True,
    ) -> Tuple[mx.array, List[mx.array]]:
        
        batch_size, seq_len = input_ids.shape

        # Gemma scales embeddings by sqrt(hidden_size)
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale

        all_hidden_states = [hidden_states] if output_hidden_states else []

        positions = mx.arange(seq_len)[None, :].astype(mx.int32)
        positions = mx.broadcast_to(positions, (batch_size, seq_len))

        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, attention_mask)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)

        return hidden_states, all_hidden_states



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
        pe: Optional[mx.array] = None,
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
            # pe: (1, seq_len, num_heads, head_dim, 2)
            # q, k: (B, seq, inner_dim) - need to reshape for RoPE then reshape back
            q = mx.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
            k = mx.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
            q, k = apply_rotary_emb_1d(q, k, pe)
            # Reshape back for attention computation
            q = mx.reshape(q, (batch_size, seq_len, -1))
            k = mx.reshape(k, (batch_size, seq_len, -1))


        q = mx.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = mx.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = mx.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        
        mask = mx.full((batch_size, seq_len, seq_len), -1e9, dtype=q.dtype)
        if attention_mask is not None:
            mask = mask + (1.0 - attention_mask[:, None, None, :].astype(q.dtype)) * -1e9

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attention_mask)
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

    def _precompute_freqs_cis(self, seq_len: int, dtype: mx.Dtype) -> mx.array:
        import math

        dim = self.num_heads * self.head_dim
        theta = self.positional_embedding_theta
        n_elem = 2 

   
        linspace_vals = mx.linspace(0.0, 1.0, dim // n_elem)
        indices = (theta ** linspace_vals) * (math.pi / 2)

        positions = mx.arange(seq_len).astype(mx.float32)
        freqs = positions[:, None] * indices[None, :]  # (seq_len, dim//2)

        cos = mx.cos(freqs)  # (seq_len, dim//2)
        sin = mx.sin(freqs)


        cos_full = mx.repeat(cos, 2, axis=-1).reshape(1, seq_len, self.num_heads, self.head_dim)
        sin_full = mx.repeat(sin, 2, axis=-1).reshape(1, seq_len, self.num_heads, self.head_dim)

        freqs_cis = mx.stack([cos_full, sin_full], axis=-1)  # (1, seq_len, num_heads, head_dim, 2)
        return freqs_cis.astype(dtype)

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
    large_val = 1e9
    x_for_min = mx.where(mask, stacked, mx.full(stacked.shape, large_val, dtype=stacked.dtype))
    x_for_max = mx.where(mask, stacked, mx.full(stacked.shape, -large_val, dtype=stacked.dtype))
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



def sanitize_gemma3_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    sanitized = {}

    for key, value in weights.items():
        new_key = None

        if key.startswith("base_text_encoder.language_model."):
            new_key = key.replace("base_text_encoder.language_model.", "")
        elif key.startswith("language_model.model."):
            new_key = key.replace("language_model.model.", "")
        elif key.startswith("language_model."):
            new_key = key.replace("language_model.", "")
        else:
            continue

        if new_key is None:
            continue

        sanitized[new_key] = value

    return sanitized


class LTX2TextEncoder(nn.Module):

    def __init__(
        self,
        model_path: str = "Lightricks/LTX-2",
        hidden_dim: int = 3840,
        num_layers: int = 49,  # 48 transformer layers + 1 embedding
    ):
        super().__init__()
        self._model_path = model_path
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Gemma 3 model
        self.config = Gemma3Config()
        self.model = Gemma3TextModel(self.config)

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

    def load(self, model_path: Optional[str] = None):
        path = model_path or self._model_path

        # Load Gemma weights from text_encoder subdirectory
        if Path(path).is_dir():
            text_encoder_path = Path(path) / "text_encoder"
            if text_encoder_path.exists():
                gemma_path = str(text_encoder_path)
            else:
                gemma_path = path
        else:
            gemma_path = path

        print(f"Loading Gemma 3 text encoder from {gemma_path}...")
        weight_files = sorted(Path(gemma_path).glob("*.safetensors"))
        all_weights = {}
        for i, wf in enumerate(weight_files):
            print(f"  Loading weight file {i+1}/{len(weight_files)}...")
            weights = mx.load(str(wf))
            all_weights.update(weights)

        # Sanitize and load Gemma weights
        sanitized = sanitize_gemma3_weights(all_weights)
        print(f"  Sanitized Gemma weights: {len(sanitized)}")
        self.model.load_weights(list(sanitized.items()), strict=False)

        # Load transformer weights for feature extractor and connector
        transformer_path = Path(model_path or self._model_path)
        transformer_files = list(transformer_path.glob("ltx-2*.safetensors"))
        if transformer_files:
            print(f"Loading transformer weights for text pipeline...")
            transformer_weights = mx.load(str(transformer_files[0]))

            # Load feature extractor (aggregate_embed)
            if "text_embedding_projection.aggregate_embed.weight" in transformer_weights:
                self.feature_extractor.aggregate_embed.weight = transformer_weights[
                    "text_embedding_projection.aggregate_embed.weight"
                ]
                print("  Loaded aggregate_embed weights")

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
                print(f"  Loaded {len(connector_weights)} connector weights")

                # Manually load learnable_registers (it's a plain mx.array, not a parameter)
                if "learnable_registers" in connector_weights:
                    self.video_embeddings_connector.learnable_registers = connector_weights["learnable_registers"]
                    print(f"  Loaded learnable_registers: {connector_weights['learnable_registers'].shape}")

        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer_path = Path(model_path or self._model_path) / "tokenizer"
        if tokenizer_path.exists():
            self.processor = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
        else:
            self.processor = AutoTokenizer.from_pretrained(gemma_path, trust_remote_code=True)
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

        # Tokenize with left padding (as in PyTorch version)
        inputs = self.processor(
            prompt,
            return_tensors="np",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])

        # Get all hidden states from Gemma
        _, all_hidden_states = self.model(input_ids, attention_mask, output_hidden_states=True)

        # Normalize and concatenate all hidden states
        concat_hidden = norm_and_concat_hidden_states(
            all_hidden_states, attention_mask, padding_side="left"
        )

        # Project through feature extractor
        features = self.feature_extractor(concat_hidden)

        # Convert attention mask to additive format for connector
        additive_mask = (attention_mask - 1).astype(features.dtype)
        additive_mask = additive_mask.reshape(attention_mask.shape[0], 1, 1, -1) * 1e9

        # Process through connector
        # Note: connector replaces padding with learnable registers and resets mask to zeros
        # This means all positions now have valid embeddings (no need for final masking)
        embeddings, _ = self.video_embeddings_connector(features, additive_mask)

        # Return embeddings without zeroing - the connector's register replacement
        # means all positions have meaningful values now
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

