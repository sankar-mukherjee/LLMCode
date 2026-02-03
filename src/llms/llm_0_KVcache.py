import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Causal Self Attention with KV Cache
# ------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        """
        x: (B, T, C)
        kv_cache: tuple(k, v) where
          k, v: (B, n_heads, T_cache, head_dim)
        """
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        att = (q @ k.transpose(-2, -1)) * self.scale

        # causal mask only when no cache (training)
        if kv_cache is None:
            T_total = k.size(2)
            mask = torch.tril(
                torch.ones(T_total, T_total, device=x.device)
            ).bool()
            att = att.masked_fill(~mask[:, -T:], float("-inf"))

        att = F.softmax(att, dim=-1)
        out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out), (k, v)


# ------------------------------------------------------------
# Feed Forward
# ------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
# Decoder Block
# ------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x

    def forward_step(self, x, kv_cache):
        attn_out, new_kv = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


# ------------------------------------------------------------
# GPT Decoder with KV Cache
# ------------------------------------------------------------
class GPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def forward_step(self, input_ids, pos, kv_cache):
        """
        input_ids: (B, 1)
        pos: scalar position index
        kv_cache: list of (k, v) per layer
        """
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        new_cache = []
        for block, layer_cache in zip(self.blocks, kv_cache):
            x, layer_cache = block.forward_step(x, layer_cache)
            new_cache.append(layer_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_cache

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        self.eval()
        B, T = input_ids.shape
        device = input_ids.device

        # initialize cache
        kv_cache = [None] * self.n_layers

        # warmup with full forward
        logits = self(input_ids)
        next_token_logits = logits[:, -1]

        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        # build cache token by token
        for t in range(max_new_tokens - 1):
            pos = torch.tensor([T + t], device=device)
            logits, kv_cache = self.forward_step(
                input_ids[:, -1:], pos, kv_cache
            )

            next_token_logits = logits[:, -1]
            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
