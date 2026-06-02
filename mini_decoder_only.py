# encoding=utf-8

import math
import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps" # 苹果GPU加速接口，Metal Performance Shaders
else:
    device = "cpu"

"""
解码器
[DEC-1] Token Embedding（词嵌入）
[DEC-2] Positional Encoding（位置编码）
[DEC-3] LayerNorm（注意力前归一化）
[DEC-4] Masked Self-Attention（带掩码的自注意力）
[DEC-5] Residual Connection（注意力残差）
[DEC-6] LayerNorm（FFN前归一化）
[DEC-7] FeedForward Network（前馈网络）
[DEC-8] Residual Connection（FFN残差）
[DEC-9] Final LayerNorm（最终归一化）
[DEC-10] Linear Projection（线性映射到词表）
[DEC-11] Softmax（概率归一化）
"""

d_k = 64
d_embedding = 512

n_heads = 8
n_layers = 6

d_ff = 2048
max_len = 512
pad_id = 0

assert d_k * n_heads == d_embedding


# [DEC-2] 位置编码
def get_sin_enc_table(n_position, embedding_dim):
    sinusoid_table = torch.zeros((n_position, embedding_dim), dtype=torch.float32)
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i / math.pow(10000, 2 * (hid_j // 2) / embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle
    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
    return sinusoid_table

# 掩码生成（为[DEC-4]服务）
def get_attn_pad_mask(seq_q, seq_k):
    B, Lq = seq_q.size()
    _, Lk = seq_k.size()
    return seq_k.eq(pad_id).unsqueeze(1).expand(B, Lq, Lk)

def get_attn_subsequent_mask(seq):
    B, L = seq.size()
    return torch.triu(torch.ones((L, L), dtype=torch.bool, device=seq.device), diagonal=1).unsqueeze(0).expand(B, L, L)

def get_decoder_mask(seq):
    pad_mask = get_attn_pad_mask(seq, seq)
    subsequent_mask = get_attn_subsequent_mask(seq)
    return (pad_mask | subsequent_mask).unsqueeze(1)

# [DEC-4核心] 缩放点积注意力
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
        scores.masked_fill_(attn_mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        return context, attn_weights

# [DEC-3 -> DEC-5] 多头自注意力模块
class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_embedding, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_embedding, d_k * n_heads, bias=False)
        self.proj = nn.Linear(n_heads * d_k, d_embedding, bias=False)
        self.layer_norm = nn.LayerNorm(d_embedding)  # [DEC-3]

    def forward(self, x, attn_mask):
        # [DEC-3] 注意力前的LayerNorm
        x_norm = self.layer_norm(x)

        # 拆分多头
        B = x.size(0)
        Q = self.W_Q(x_norm).view(B, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(x_norm).view(B, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(x_norm).view(B, -1, n_heads, d_k).transpose(1, 2)
        attn_mask = attn_mask.expand(-1, n_heads, -1, -1)

        # [DEC-4] Masked Self-Attention
        context, attn_weights = ScaledDotProductAttention()(Q, K, V, attn_mask)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(B, -1, n_heads * d_k)
        output = self.proj(context)

        # [DEC-5] 注意力残差连接
        output = x + output
        return output, attn_weights

# [DEC-6 -> DEC-8] 前馈网络模块
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(d_embedding, d_ff)
        self.linear2 = nn.Linear(d_ff, d_embedding)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_embedding)  # [DEC-6]

    def forward(self, x):
        # [DEC-6] FFN前的LayerNorm
        x_norm = self.layer_norm(x)

        # [DEC-7] FeedForward Network
        output = self.linear2(self.gelu(self.linear1(x_norm)))

        # [DEC-8] FFN残差连接
        output = x + output
        return output

# 解码器块（堆叠注意力和FFN）
class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention()  # [DEC-3 -> DEC-5]
        self.ffn = FeedForward()  # [DEC-6 -> DEC-8]

    def forward(self, x, attn_mask):
        x, attn_weights = self.self_attn(x, attn_mask)
        x = self.ffn(x)
        return x, attn_weights

# [DEC-1 -> DEC-11完整流程] Decoder Only模型
class DecoderOnlyLM(nn.Module):
    def __init__(self, vocab_size, tie_weights=True):
        super().__init__()
        # [DEC-1] Token Embedding
        self.tok_emb = nn.Embedding(vocab_size, d_embedding, padding_idx=pad_id)
        # [DEC-2] Positional Encoding
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sin_enc_table(max_len, d_embedding), freeze=True
        )
        # 多层解码器块
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(n_layers)])
        # [DEC-9] Final LayerNorm
        self.final_ln = nn.LayerNorm(d_embedding)
        # [DEC-10] Linear Projection
        self.lm_head = nn.Linear(d_embedding, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.max_len = max_len
        self.pad_id = pad_id

    def forward(self, idx):
        B, L = idx.size()
        device = idx.device

        # 输入截断
        if L > self.max_len:
            idx = idx[:, -self.max_len:]
            L = self.max_len

        # [DEC-1] Token Embedding
        tok_emb = self.tok_emb(idx)

        # [DEC-2] Positional Encoding
        pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_emb(pos_idx)

        # 合并嵌入
        x = tok_emb + pos_emb

        # 获取掩码
        attn_mask = get_decoder_mask(idx)

        # 多层Block处理
        attn_weights_list = []
        for block in self.blocks:
            x, attn_weights = block(x, attn_mask)
            attn_weights_list.append(attn_weights)

        # [DEC-9] Final LayerNorm
        x = self.final_ln(x)

        # [DEC-10] Linear Projection
        logits = self.lm_head(x)

        # [DEC-11] Softmax（注意：训练时通常在损失函数中合并softmax，这里显式保留步骤）
        probs = torch.softmax(logits, dim=-1)

        return logits, probs, attn_weights_list

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=80, eos_id=None, temperature=0.8):
        self.eval()
        for _ in range(max_new_tokens):
            current_len = idx.size(1)
            if current_len >= self.max_len:
                break

            # 前向传播到[DEC-11]
            logits, _, _ = self(idx)
            next_logits = logits[:, -1, :] / temperature  # 温度缩放

            # [DEC-11] 显式softmax采样
            next_probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(next_probs, 1)

            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None and idx.size(0) == 1 and int(next_id[0]) == eos_id:
                break

        self.train()
        return idx


__all__ = ["DecoderOnlyLM", "device", "pad_id"]



