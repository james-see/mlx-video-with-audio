import mlx.core as mx
import mlx.nn as nn


class PixArtAlphaTextProjection(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int | None = None,
        bias: bool = True,
        act_fn: str = "gelu_tanh",
    ):
        
        super().__init__()

        out_features = out_features or hidden_size
        self.linear1 = nn.Linear(in_features, hidden_size, bias=bias)
        if act_fn == "gelu_tanh":
            self.act = nn.GELU(approx="tanh")  
        elif act_fn == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear2 = nn.Linear(hidden_size, out_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x
