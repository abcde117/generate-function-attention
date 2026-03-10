

from ori_model import *
from models import GFAPOBLOCK, GFAPFBLOCK, GFAPOKBLOCK, GFAPFKBLOCK



class BaseMiniLM(nn.Module):
  def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        rope_theta: float,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        assert d_model % num_heads == 0
        d_head = d_model // num_heads
        self.positional_encoder = RotaryEmbedding(
            max_seq_len=context_length,
            d_k=d_head,
            theta=rope_theta
        )


        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.layers = nn.ModuleList([
           BaseTransformerBlock(d_model,num_heads,d_ff=int(8/3*d_model),positional_encoder=self.positional_encoder,mask=True) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

  def forward(self, x):


# Token embedding: (B, T, d_model)
        x = self.token_embeddings(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # LN + LM head
        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits
  


class GFAPMiniLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        context_length: int,
        *,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        self.context_length = context_length
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        if use_rope:
            self.positional_encoder = RotaryEmbedding(
                max_seq_len=context_length,
                d_k=d_model,
                theta=rope_theta
            )
        else:
            self.positional_encoder = None

        d_ff = int(8 / 3 * d_model)

        self.layers = nn.ModuleList([
            self.build_block(
                d_model=d_model,
                d_ff=d_ff,
                mask=True,
                positional_encoder=self.positional_encoder
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def build_block(self, d_model, d_ff, mask, positional_encoder):
        raise NotImplementedError

    def forward(self, x):
        assert x.size(1) <= self.context_length, \
            f"sequence length {x.size(1)} > context_length {self.context_length}"

        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        return self.lm_head(x)
    
class GFAPMiniLM_PO(GFAPMiniLM):
    def build_block(self, d_model, d_ff, mask, positional_encoder):
        return GFAPOBLOCK(d_model, d_ff, mask, positional_encoder)


class GFAPMiniLM_PF(GFAPMiniLM):
    def build_block(self, d_model, d_ff, mask, positional_encoder):
        return GFAPFBLOCK(d_model, d_ff, mask, positional_encoder)


class GFAPMiniLM_POK(GFAPMiniLM):
    def build_block(self, d_model, d_ff, mask, positional_encoder):
        return GFAPOKBLOCK(d_model, d_ff, mask, positional_encoder)


class GFAPMiniLM_PFK(GFAPMiniLM):
    def build_block(self, d_model, d_ff, mask, positional_encoder):
        return GFAPFKBLOCK(d_model, d_ff, mask, positional_encoder)
