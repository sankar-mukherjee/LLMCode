import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAttention(nn.Module):
    def __init__(self, d_model, n_head, max_seq_len, dropout):
        super().__init__()
        assert d_model % n_head == 0

        self.max_seq_len = max_seq_len
        self.n_head = n_head
        self.h_dim = d_model // n_head
        self.scale = self.h_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3*d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        )
        
    
    def forward(self, x, past_kv=None):
        B, T, C = x.shape
        assert T <= self.max_seq_len, "max limit reached"

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.h_dim).transpose(1,2)
        k = k.view(B, T, self.n_head, self.h_dim).transpose(1,2)
        v = v.view(B, T, self.n_head, self.h_dim).transpose(1,2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_kv = (k, v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        total_len = k.size(-2)
        mask = self.causal_mask[:T, :total_len].to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(~mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), present_kv


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, max_seq_len, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalAttention(d_model, n_head, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x, past_kv=None):
        attn_out, present_kv = self.attn(self.ln1(x), past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, present_kv
    

class GPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, max_seq_len, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.text_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_head, max_seq_len, dropout)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)


    def forward(self, input_ids, past_kvs=None):
        B, T = input_ids.shape

        pos = torch.arange(T, device=input_ids.device)

        x = self.text_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        new_kvs = []

        for i, block in enumerate(self.blocks):
            past_kv = None if past_kvs is None else past_kvs[i]
            x, present_kv = block(x, past_kv)
            new_kvs.append(present_kv)

        logits = self.lm_head(self.ln_f(x))
        return logits, new_kvs
    
    @torch.no_grad()
    def generate(self, input_ids, max_tokens, temp=0.1):
        self.eval()
        past_kvs = None
        # first forward pass
        logits, past_kvs = self(input_ids, past_kvs)

        for _ in range(max_tokens):
            # input_cond = input_ids[:, -self.max_seq_len:]
            # logits = self(input_cond)
            next_token_logits = logits[:,-1,:]

            if temp == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temp, dim=-1)
                next_token = torch.multinomial(probs, 1)

            input_ids = torch.concat([input_ids, next_token], dim=1)
            logits, past_kvs = self(next_token, past_kvs)

        return input_ids



