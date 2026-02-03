import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        # Ensure embedding dimension can be evenly split across heads
        assert d_model % n_heads == 0

        # Number of attention heads
        self.n_heads = n_heads
        # Dimensionality of each attention head
        self.head_dim = d_model // n_heads
        # Scaling factor for dot-product attention (prevents large logits)
        self.scale = self.head_dim ** -0.5

        # Single linear layer to compute queries, keys, and values together
        # Output shape: (B, T, 3 * d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # Output projection after attention
        self.out = nn.Linear(d_model, d_model)

        # Dropout applied to attention weights
        self.attn_dropout = dropout
        # Dropout applied to the final output projection
        self.resid_dropout = nn.Dropout(dropout)


    def forward(self, x):
        # x shape: (B, T, C)
        # B = batch size, T = sequence length, C = embedding dimension
        B, T, C = x.shape

        # Ensure sequence length does not exceed mask capacity
        assert T <= self.causal_mask.size(0)

        # Compute query, key, and value projections in one matmul
        # qkv shape: (B, T, 3C)
        qkv = self.qkv(x)

        # Split into q, k, v each of shape (B, T, C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape and transpose to bring heads forward
        # Final shape: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True
        )

        # Merge heads back together
        # Shape: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection and residual dropout
        return self.resid_dropout(self.out(out))


# FeedForward Optimization: SwiGLU
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model, 8 * d_model // 3)
        self.w2 = nn.Linear(d_model, 8 * d_model // 3)
        self.w3 = nn.Linear(8 * d_model // 3, d_model)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, dropout=0.1):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)


    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device

        assert T <= self.max_seq_len, "Sequence too long"

        pos = torch.arange(T, device=device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=1.0,
    ):
        self.eval()

        for _ in range(max_new_tokens):
            input_cond = input_ids[:, -self.max_seq_len:]
            logits = self(input_cond)
            next_token_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
