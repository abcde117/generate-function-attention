
import torch
from einops import einsum ,rearrange, repeat
import jaxtyping
from  jaxtyping  import Array,Float,Int,Bool
from torch import nn
import einx
import torch.nn.functional as F
import math




class Embedding(nn.Module):
    def __init__(self,num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim), std=1.0,
                                 a=-3,b=3))

    def forward(self,token_ids:Int[Array,"..."])->Float[Array,"... d_model"]:
        return self.weight[token_ids,:]

class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self,x:Float[Array,"batch_size seq_len d_model"])->Float[Array,"batch_size ..."]:
        in_dtype=  x.dtype
        x=x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        return self.weight * (x *rms)
class RotaryEmbedding(nn.Module):
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache",
            RotaryEmbedding._init_cache(max_seq_len, d_k, theta), persistent=False
        )
    @staticmethod
    def _init_cache(max_seq_len:int, d_k:int, theta:float)-> Float[Array,"2 max_seq_len d_k/2"]:
        assert d_k % 2 == 0, "d_k must be even"
        d=torch.arange(0,d_k,2).float()
        t=torch.arange(0,max_seq_len).float()
        freqs= theta ** (-d / d_k)
        freqs= einsum(t,freqs,'t,f -> t f')
        cos,sin= torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos,sin))

    def forward(self,x:Float[Array,"... seq  d_k"],pos_ids: Int[Array, " ... seq"])->Float[Array,"...  seq d_k"]:
        x1, x2 = rearrange(x, "... (d r) -> ... d r", r=2).unbind(-1)
        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._freq_cis_cache, pos_ids)
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()
        return result
    
    
    
def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn .Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.w3 = nn.Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))
    
class UniversalClassifier(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out)
        )

    def forward(self, x):
        x = x.mean(dim=1)  # mean over tokens
        return self.classifier(x)