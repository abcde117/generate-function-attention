
import torch
from einops import einsum ,rearrange, repeat
import jaxtyping
from  jaxtyping  import Array,Float,Int,Bool
from torch import nn
import einx
from abc import ABC, abstractmethod
import torch.fft as fft
import torch.nn.functional as F
import math

from uni_func import  RMSNorm,RotaryEmbedding,SwiGLU,silu


def gen_func_proj_ori(q,k,v,mask):

  '''
  B, n, m = q.shape
  K=2*m-1
  q_pad = F.pad(q, (m-1, m-1))
  q_win =q_pad.unfold(-1, m, 1)
  out=einsum(q_win, k,'... i k t, ... j t -> ... i j k')'''
  B, n, m = q.shape
  K = 2*m - 1
  outer = q[:, :, None, :, None] * k[:, None, :, None, :]
  idx = torch.arange(m, device=q.device)
  idx_sum = idx[:, None] + idx[None, :]
  out = torch.zeros(B, n, n, K, device=q.device)
  outer_flat = outer.view(B, n, n, -1)
  idx_flat = idx_sum.reshape(-1)

  outer_flat = outer.view(B, n, n, -1)        # [B, n, n, m*m]
  idx_flat = idx_sum.reshape(-1)               # [m*m]

  out.scatter_add_(
    dim=-1,
    index=idx_flat[None, None, None, :].expand(B, n, n, -1),
    src=outer_flat
)




  d = torch.arange(2, 2*m+1, device=q.device)
  nd = torch.cat([torch.arange(1, d[-1]//2+1, device=q.device),
                    torch.arange(d[-1]//2-1, 0, -1, device=q.device)])
  #scaler = torch.sqrt(d / nd)
  scaler = d / nd
  #score=out*scaler[None,None,None,:]
  score=out*scaler[None,None,None,:]
  #score=out/nd
  #score=out/torch.sqrt(nd)
  #score=F.tanh(score)
  #score=F.gelu(score)
  score=F.log_softmax(score, dim=-1)
  if mask is not None:


    mask= rearrange( mask,'b a i j -> b i j a')


    score=torch.where(mask, score, torch.tensor(float(0)))
  o=einsum(F.log_softmax(score, dim=-1),v,'... i j k, ... j l->... i k l')
  o=einsum(o, '... i k l -> ... i l')

  return o

def gen_func_proj_fft(q,k,v,mask):

  B, n, m = q.shape
  K=2*m-1

  q_f = fft.rfft(q, n=K,dim=-1)
  k_f = fft.rfft(k, n=K,dim=-1)

  out_f = einsum(q_f, k_f, "... i f, ... j f -> ... i j f")
  out = fft.irfft(out_f, n=K,dim=-1)

  d = torch.arange(2, 2*m + 1, device=q.device)
  nd = torch.cat([
        torch.arange(1, d[-1]//2 + 1, device=q.device),
        torch.arange(d[-1]//2 - 1, 0, -1, device=q.device)
    ])
  #scaler = torch.sqrt(d / nd)   # [K]
  scaler = d / nd
  #score=out*scaler[None,None,None,:]
  score = out * scaler[None, None, None, :]   # [B, n, n, K]
  #score=out/torch.sqrt(nd)
  #score=F.tanh(score)
  #score=F.gelu(score)
  score = F.log_softmax(score, dim=-1)
  if mask is not None:

    mask=  rearrange( mask,'b a i j -> b i j a')


    score=torch.where(mask, score, torch.tensor(float(0)))

  o = einsum(score, v, "... i j k, ... j l -> ... i k l")


  o = einsum(o, "... i k l -> ... i l")

  return o



def gen_func_proj_ori_k(q,k,v,mask):

  '''
  B, n, m = q.shape
  K=2*m-1
  q_pad = F.pad(q, (m-1, m-1))
  q_win =q_pad.unfold(-1, m, 1)
  out=einsum(q_win, k,'... i k t, ... j t -> ... i j k')'''
  B, n, m = q.shape
  K = 2*m - 1
  outer = q[:, :, None, :, None] * k[:, None, :, None, :]
  idx = torch.arange(m, device=q.device)
  idx_sum = idx[:, None] + idx[None, :]
  out = torch.zeros(B, n, n, K, device=q.device)
  outer_flat = outer.view(B, n, n, -1)
  idx_flat = idx_sum.reshape(-1)

  outer_flat = outer.view(B, n, n, -1)        # [B, n, n, m*m]
  idx_flat = idx_sum.reshape(-1)               # [m*m]

  out.scatter_add_(
    dim=-1,
    index=idx_flat[None, None, None, :].expand(B, n, n, -1),
    src=outer_flat
)




  d = torch.arange(2, 2*m+1, device=q.device)
  nd = torch.cat([torch.arange(1, d[-1]//2+1, device=q.device),
                    torch.arange(d[-1]//2-1, 0, -1, device=q.device)])
  scaler = torch.sqrt( nd)
  #xk=torch.sqrt(d / nd)
  xk=d/nd

  score=out/scaler[None,None,None,:]
  score=score*xk[None,None,None,:]
  #score=F.tanh(score)
  #score=F.gelu(score)
  score=F.log_softmax(score, dim=-2)


  if mask is not None:


    mask= rearrange( mask,'b a i j -> b i j a')


    score=torch.where(mask, score, torch.tensor(float(0)))
  o=einsum(F.log_softmax(score, dim=-1),v,'... i j k, ... j l->... i k l')
  o=einsum(o, '... i k l -> ... i l')

  return o



def gen_func_proj_fft_k(q,k,v,mask):

  B, n, m = q.shape
  K=2*m-1

  q_f = fft.rfft(q, n=K,dim=-1)
  k_f = fft.rfft(k, n=K,dim=-1)

  out_f = einsum(q_f, k_f, "... i f, ... j f -> ... i j f")
  out = fft.irfft(out_f, n=K,dim=-1)

  d = torch.arange(2, 2*m + 1, device=q.device)
  nd = torch.cat([
        torch.arange(1, d[-1]//2 + 1, device=q.device),
        torch.arange(d[-1]//2 - 1, 0, -1, device=q.device)
    ])
  scaler = torch.sqrt( nd)   # [K]
  #xk=torch.sqrt(d / nd)
  xk=d/nd

  score = out / scaler[None, None, None, :]
  score=score *xk[None, None, None, :]
  #score=F.tanh(score)
  #score=F.gelu(score)
  score = F.log_softmax(score, dim=-2)
  if mask is not None:

    mask=  rearrange( mask,'b a i j -> b i j a')


    score=torch.where(mask, score, torch.tensor(float(0)))

  o = einsum(score, v, "... i j k, ... j l -> ... i k l")


  o = einsum(o, "... i k l -> ... i l")

  return o



class GenFuncProA(nn.Module):
    def __init__(self, d_model, d_k, mask=False, positional_encoder=None):
        super().__init__()
        self.wq = nn.Linear(d_model, d_k)
        self.wk = nn.Linear(d_model, d_k)
        self.wv = nn.Linear(d_model, d_k)
        self.proj = nn.Linear(d_k, d_k)

        self.mask = mask
        self.positional_encoder = positional_encoder

    def apply_positional(self, q, k, pos_1d):
        if self.positional_encoder is not None:
            q = self.positional_encoder(q, pos_1d)
            k = self.positional_encoder(k, pos_1d)
        return q, k

    def build_causal_mask(self, pos_1d, batch_shape):
        """
        pos_1d: (S,)
        return: (B..., Q, K)
        """
        qi = einx.rearrange(
            "s -> b... 1 s 1",
            pos_1d,
            b=[1] * len(batch_shape)
        )
        kj = einx.rearrange(
            "s -> b... 1 1 s",
            pos_1d,
            b=[1] * len(batch_shape)
        )
        return qi >= kj



class GFAPO(GenFuncProA):
    def forward(self, x):
        *b, seq_len, _ = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        pos = torch.arange(seq_len, device=x.device)

        q, k = self.apply_positional(q, k, pos)

        causal_mask = None
        if self.mask:
            causal_mask = self.build_causal_mask(pos, b)

        p = gen_func_proj_ori(q, k, v, causal_mask)
        return self.proj(p)


class GFAPF(GenFuncProA):
    def forward(self, x):
        *b, seq_len, _ = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        pos = torch.arange(seq_len, device=x.device)

        q, k = self.apply_positional(q, k, pos)

        causal_mask = None
        if self.mask:
            causal_mask = self.build_causal_mask(pos, b)

        p = gen_func_proj_fft(q, k, v, causal_mask)
        return self.proj(p)
    
    

class GFAPOK(GenFuncProA):
  def forward(self, x):
        *b, seq_len, _ = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        pos = torch.arange(seq_len, device=x.device)

        q, k = self.apply_positional(q, k, pos)

        causal_mask = None
        if self.mask:
            causal_mask = self.build_causal_mask(pos, b)

        p = gen_func_proj_ori_k(q, k, v, causal_mask)
        return self.proj(p)


class GFAPFK(GenFuncProA):
    def forward(self, x):
        *b, seq_len, _ = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        pos = torch.arange(seq_len, device=x.device)

        q, k = self.apply_positional(q, k, pos)

        causal_mask = None
        if self.mask:
            causal_mask = self.build_causal_mask(pos, b)

        p = gen_func_proj_fft_k(q, k, v, causal_mask)
        return self.proj(p)

class GFABLOCK(nn.Module, ABC):
    def __init__(self, d_model: int, d_ff: int,mask:bool, positional_encoder=None):
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

        self.attn = self.build_attn(d_model,mask,positional_encoder=positional_encoder)
        self.ffn  = self.build_ffn(d_model, d_ff)


    @abstractmethod
    def build_attn(self, d_model,mask,positional_encoder):
        pass

    @abstractmethod
    def build_ffn(self, d_model, d_ff):
        pass

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
    
class GFAPOBLOCK(GFABLOCK):
    def build_attn(self, d_model, mask, positional_encoder):
        return GFAPO(
            d_model,
            d_model,
            mask=mask,
            positional_encoder=positional_encoder
        )

    def build_ffn(self, d_model, d_ff):
        return SwiGLU(d_model, d_ff)

class GFAPFBLOCK(GFABLOCK):
    def build_attn(self, d_model, mask, positional_encoder):
        return GFAPF(
            d_model,
            d_model,
            mask=mask,
            positional_encoder=positional_encoder
        )

    def build_ffn(self, d_model, d_ff):
        return SwiGLU(d_model, d_ff)

class GFAPOKBLOCK(GFABLOCK):
    def build_attn(self, d_model, mask, positional_encoder):
        return GFAPOK(
            d_model,
            d_model,
            mask=mask,
            positional_encoder=positional_encoder
        )

    def build_ffn(self, d_model, d_ff):
        return SwiGLU(d_model, d_ff)



class GFAPFKBLOCK(GFABLOCK):
    def build_attn(self, d_model, mask, positional_encoder):
        return GFAPFK(
            d_model,
            d_model,
            mask=mask,
            positional_encoder=positional_encoder
        )

    def build_ffn(self, d_model, d_ff):
        return SwiGLU(d_model, d_ff)
    
    
    
class MiniVT(nn.Module, ABC):
    def __init__(
        self,
        d_in=1,
        d_out=8,
        d_model=16,
        patch_size=4,
        img_size=32,
        depth=4,
    ):
        super().__init__()

        # Patch Embedding
        self.initconv = nn.Conv2d(
            d_in, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positions = nn.Embedding(num_patches + 1, d_model)
        self.d_model=d_model
        self.d_ff=int(8/3*d_model)

        self.blocks = nn.ModuleList([
            self.build_block(d_model,self.d_ff,mask=False,positional_encoder=None)
            for _ in range(depth)
        ])

        self.proj = nn.Linear(d_model, d_out)

    @abstractmethod
    def build_block(self, d_model,d_ff,mask=False):
        pass

    def forward(self, x):
        x = self.initconv(x)
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.positions(pos_ids)

        for block in self.blocks:
            x = block(x)

        return self.proj(x)
    
    
    
class MiniVT_PO(MiniVT):
    def build_block(self, d_model,d_ff,mask,positional_encoder):
        return GFAPOBLOCK(d_model,d_ff,mask,positional_encoder)

class MiniVT_PF(MiniVT):
    def build_block(self, d_model,d_ff,mask,positional_encoder):
        return GFAPFBLOCK(d_model,d_ff,mask,positional_encoder)
class MiniVT_POK(MiniVT):
    def build_block(self, d_model,d_ff,mask,positional_encoder):
        return GFAPOKBLOCK(d_model,d_ff,mask,positional_encoder)

class MiniVT_PFK(MiniVT):
    def build_block(self, d_model,d_ff,mask,positional_encoder):
        return GFAPFKBLOCK(d_model,d_ff,mask,positional_encoder)
