

import torch
from einops import einsum ,rearrange, repeat
import jaxtyping
from  jaxtyping  import Array,Float,Int,Bool
from torch import nn
import einx
import torch.nn.functional as F
import math

from uni_func import  RMSNorm,RotaryEmbedding,SwiGLU,silu



def scale_dot_product_attention(
    Query:Float[Array,"... queries d_k"],
    Key:Float[Array,"... keys d_k"],
    Value:Float[Array,"... values d_v"],
    mask:Bool[Array,"... queries keys"] | None = None,
)->Float[Array,"... queries d_v"]:
    d_k=Key.shape[-1]
    attention_score=einsum(Query,Key,"... queries d_k,... keys d_k-> ... queries keys")/math.sqrt(d_k)
    if mask is not None:
        attention_score=torch.where(mask, attention_score, torch.tensor(float('-inf')))
    attention_weights=torch.softmax(attention_score,dim=-1)

    return einsum(attention_weights,Value,"... query key, ... key d_v ->  ... query d_v")



class BaseAttentionBlock(nn.Module):

   def __init__(
       self,
       d_model:int,
       num_heads:int,
       positional_encoder:None,
       mask:Bool[Array,"... queries keys"] | None = None,
   ):
       super().__init__()

       self.d_model = d_model
       self.num_heads = num_heads
       self.d_k = d_model // num_heads
       self.d_v=self.d_k
       self.mask=mask

       self.q_proj=nn.Linear(self.d_model,self.num_heads*self.d_k)
       self.k_proj=nn.Linear(self.d_model,self.num_heads*self.d_k)
       self.v_proj=nn.Linear(self.d_model,self.num_heads*self.d_v)

       self.out_proj=nn.Linear(self.num_heads*self.d_v,self.d_model)

       self.positional_encoder = positional_encoder


   def forward(
        self,
        x: Float[Array, "... sequence d_k"],
        token_positions: Int[Array, "... seq"] | None = None
    ) -> Float[Array, "... sequence d_v"]:

        *b,seq_len,d_model=x.shape

        assert d_model ==self.d_model
        Q=self.q_proj(x)
        K=self.k_proj(x)
        V=self.v_proj(x)

        Q,K,V=(
            rearrange(X,"... sequence (heads d_k)->... heads sequence d_k" ,heads=self.num_heads)
            for X in (Q, K, V)
        )

        causal_mask =None
        if self.mask :
          seq = torch.arange(seq_len, device=x.device)
          qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
          kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
          causal_mask = qi >= kj

        if token_positions is  None:
            token_positions = einx.rearrange("seq -> b... seq",torch.arange(seq_len, device=x.device),b=[1] * len(b))
        token_positions=rearrange(token_positions, "... seq -> ... 1 seq ")
        if self.positional_encoder is not None:
         Q=self.positional_encoder(Q,token_positions)
         K=self.positional_encoder(K,token_positions)


        attn_output= scale_dot_product_attention(
            Q, K, V, mask=causal_mask
        )

        attn_output=rearrange(attn_output,"... h seq d_v -> ... seq (h d_v)").contiguous()
        output=self.out_proj(attn_output)
        return output
class BaseTransformerBlock(nn.Module):
   def __init__(self,d_model:int,num_heads,  d_ff: int,positional_encoder:None,mask=False):


     super().__init__()
     self.d_model = d_model
     self.num_heads = num_heads
     self.d_k = d_model // num_heads
     self.d_v=self.d_k
     #self.out_proj=nn.Linear(self.num_heads*self.d_v,self.d_model)
     self.ln1 = RMSNorm(d_model)
     self.ln2 = RMSNorm(d_model)
     self.positional_encoder = positional_encoder
     self.attn = BaseAttentionBlock(d_model,num_heads,positional_encoder,mask=mask)
     self.ffn  = SwiGLU(d_model,d_ff)



   def forward(self, x:Float[Array,"... seq_len d_model"])->Float[Array,"... seq_len d_model"]:
      x = x + self.attn(self.ln1(x))
      x = x + self.ffn(self.ln2(x))
      return x
  


class MiniVT_ORI(nn.Module):
   def __init__(
        self,
        d_in=1,
        d_out=8,
        d_model=16,
        patch_size=4,
        num_heads=2,
        img_size=32,
        depth=4,
        positional_encoder=None
    ):
        super().__init__()

        # ----- Patch Embedding -----
        self.initconv = nn.Conv2d(
            d_in, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positions = nn.Embedding(num_patches + 1, d_model)

        d_k = d_model // num_heads


        self.blocks = nn.ModuleList([
           BaseTransformerBlock(d_model,num_heads,d_ff=int(8/3*d_model),positional_encoder=positional_encoder,mask=False) for _ in range(depth)
        ])

        self.proj = nn.Linear(d_model, d_out)
   def forward(self, x):
        # Patch Embedding
        x = self.initconv(x)
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")

        # CLS + Position
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        pos_ids = torch.arange(x.size(1), device=x.device).reshape(1, -1)
        x = x + self.positions(pos_ids)


        for block in self.blocks:
            x = block(x)



        return self.proj(x)