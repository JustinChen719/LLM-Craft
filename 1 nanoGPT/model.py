import torch
import torch.nn as nn
import torch.nn.functional as F

from constant import *


class Attention(nn.Module):
    def __init__(self, head_dim, dropout=0.1):
        super().__init__()
        # q k v 权重
        # 注意这里只是一种多头注意力的写法（先计算后合并）
        self.Q = nn.Linear(embd_dim, head_dim, bias=False)
        self.K = nn.Linear(embd_dim, head_dim, bias=False)
        self.V = nn.Linear(embd_dim, head_dim, bias=False)
        # mask
        self.register_buffer("mask", torch.tril(torch.ones(context_size, context_size)))
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # 计算qkv
        q, k, v = self.Q(x), self.K(x), self.V(x)  # (B, T, C) -> (B, T, H)

        x = (q @ k.transpose(-1, -2)) * C ** -0.5  # (B, T, T)
        x = x.masked_fill(mask=self.mask[:T, :T] == 0, value=float("-inf"))
        x = F.softmax(x, dim=-1)
        x = self.dropout(x)
        x = x @ v  # (B, T, H)

        return x  # (B, T, H)


class MultiAttention(nn.Module):
    def __init__(self, n_head, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList(Attention(embd_dim // n_head) for _ in range(0, n_head))
        self.proj = nn.Linear(embd_dim, embd_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, H) * n_head -> (B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out  # (B, T, C)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embd_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embd_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.ffn(x)
        return out


class Block(nn.Module):
    def __init__(self, n_head, dropout=0.1):
        super().__init__()
        self.ma = MultiAttention(n_head, dropout)
        self.ln1 = nn.LayerNorm(embd_dim)
        self.ffn = FeedForward(embd_dim * 4, dropout)
        self.ln2 = nn.LayerNorm(embd_dim)

    def forward(self, x):
        # 注意这里使用的是pre-norm，还有一种是post-norm，注意区分
        x = x + self.ma(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, dropout=0.1):
        super().__init__()
        # embedding表和位置嵌入表
        self.token_embedding_table = nn.Embedding(vocab_size, embd_dim)
        self.position_embedding_table = nn.Embedding(context_size, embd_dim)

        self.blocks = nn.Sequential(*[Block(n_head, dropout) for _ in range(0, n_layer)])
        self.ln = nn.LayerNorm(embd_dim)

        self.liner = nn.Linear(embd_dim, vocab_size)

    def forward(self, idx, y=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.liner(x)
        if y is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(logits, y)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
