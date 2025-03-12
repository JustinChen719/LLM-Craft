import torch
import torch.nn as nn
import torch.nn.functional as F

from constant import *


class Attention(nn.Module):
    def __init__(self, head_dim, dropout=0.1):
        super().__init__()
        # QKV 线性变换层（无偏置项）
        self.Q = nn.Linear(embd_dim, head_dim, bias=False)  # 输入维度embd_dim → 头维度head_dim
        self.K = nn.Linear(embd_dim, head_dim, bias=False)
        self.V = nn.Linear(embd_dim, head_dim, bias=False)

        # 三角掩码（用于自注意力的因果掩码）
        self.register_buffer("mask", torch.tril(torch.ones(context_size, context_size)))  # 下三角掩码

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # B=批量大小, T=时间步长, C=嵌入维度

        # 计算QKV向量
        q, k, v = self.Q(x), self.K(x), self.V(x)  # (B, T, head_dim)

        # Attention计算
        weights = q @ k.transpose(-1, -2)  # 点积注意力分数 (B, T, T)
        weights *= C ** -0.5  # 缩放因子（防止梯度爆炸）

        # 应用掩码（将未来位置置为-inf）
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # 计算softmax得到权重分布
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)  # Dropout

        # 加权求和得到输出
        out = weights @ v  # (B, T, head_dim)

        return out


class MultiAttention(nn.Module):
    def __init__(self, n_head, dropout=0.1):
        super().__init__()
        head_dim = embd_dim // n_head  # 每个头的维度

        # 创建多头注意力层列表
        self.heads = nn.ModuleList([
            Attention(head_dim, dropout) for _ in range(n_head)
        ])

        # 输出投影层（将多头结果合并为原始维度）
        self.proj = nn.Linear(embd_dim, embd_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 并行计算所有头的注意力输出
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, embd_dim)

        # 投影合并并添加Dropout
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        # 前馈网络结构：Linear -> ReLU -> Linear -> Dropout
        self.ffn = nn.Sequential(
            nn.Linear(embd_dim, hidden_dim),  # 输入到隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, embd_dim),  # 隐藏层到输出
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class Block(nn.Module):
    def __init__(self, n_head, dropout=0.1):
        super().__init__()
        # 多头注意力层
        self.ma = MultiAttention(n_head, dropout)
        self.ln1 = nn.LayerNorm(embd_dim)  # 第一个层归一化

        # 前馈网络层
        self.ffn = FeedForward(embd_dim * 4, dropout)  # 隐藏层维度通常是4倍embd_dim
        self.ln2 = nn.LayerNorm(embd_dim)  # 第二个层归一化

    def forward(self, x):
        # 使用Pre-Normalization（先归一化后计算）
        # 多头注意力分支
        x = x + self.ma(self.ln1(x))  # 残差连接
        # 前馈网络分支
        x = x + self.ffn(self.ln2(x))  # 残差连接
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, dropout=0.1):
        super().__init__()
        # 词嵌入和位置嵌入层
        self.token_embedding_table = nn.Embedding(vocab_size, embd_dim)  # 词表到嵌入空间
        self.position_embedding_table = nn.Embedding(context_size, embd_dim)  # 位置编码
        # transformer块堆叠
        self.blocks = nn.Sequential(*[
            Block(n_head, dropout) for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(embd_dim)  # 最终归一化层
        # 预测层（将嵌入维度映射到词表大小）
        self.liner = nn.Linear(embd_dim, vocab_size)

    def forward(self, idx, y=None):
        B, T = idx.shape
        device = idx.device

        # 嵌入层组合（词嵌入+位置嵌入）
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb

        # 通过所有Block层
        x = self.blocks(x)
        x = self.ln(x)  # 最终归一化

        # 预测层输出logits
        logits = self.liner(x)

        # 计算损失（如果标签y存在）
        if y is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # 展平为(B*T, C)
            y = y.view(B * T)  # 展平为(B*T)
            loss = F.cross_entropy(logits, y)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """生成文本的推理函数"""
        for _ in range(max_new_tokens):
            # 截断输入到context_size长度（防止超出位置编码范围）
            idx_cond = idx[:, -context_size:]

            # 前向计算得到最后一个token的logits
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # 取最后一个时间步

            # 生成概率分布并采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # 随机采样

            # 拼接新生成的token
            idx = torch.cat((idx, idx_next), dim=1)

        return idx  # 返回完整序列（包含原始输入和生成部分）
