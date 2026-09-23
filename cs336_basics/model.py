import torch
from torch import nn
import math

def init_weights(m:nn.Module):
    if isinstance(m, Linear):
        std = math.sqrt(2.0 / (m.d_in + m.d_out))
        nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-3*std, b=3*std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, Embedding):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    # elif isinstance(m, RMSNorm):
    #     nn.init.ones_(m.weight)


# uv run pytest -k test_linear
class Linear(nn.Module):
    """ A simple linear layer implementated from scratch. """
    d_in: int
    d_out: int
    weight: torch.Tensor

    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty((d_out, d_in), **kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(d_out, **kwargs))
        else:
            self.register_parameter("bias", None)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [..., in_features] -> [..., out_features]
        return torch.einsum('...i,oi->...o', x, self.weight) + (self.bias if self.bias is not None else 0)

# uv run pytest -k test_embedding
class Embedding(nn.Module):
    """ A simple embedding layer implemented from scratch. """
    num_embeddings: int
    embedding_dim: int
    weight: torch.Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
    ) -> None:
        kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **kwargs))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # [...] -> [..., embedding_dim]
        return self.weight[token_ids]