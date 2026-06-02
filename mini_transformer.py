# encoding=utf-8

import numpy as np
import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps" # 苹果GPU加速接口，Metal Performance Shaders
else:
    device = "cpu"

"""
编码器
[ENC-1] Embedding
[ENC-2] Positional Encoding（当前使用“绝对正弦”；相对/旋转在注意力里改 Q/K）
[ENC-3] Multi-Head Attention（Q/K/V 映射 + 注意力矩阵）
[ENC-4] FFN（两层非线性；用 1×1 Conv 实现的等价形式）
[ENC-5] Residual + LayerNorm

解码器
[DEC-1] Masked Self-Attention（PAD + Subsequent 上三角）
[DEC-2] Cross-Attention（Q 来自解码器，K/V 来自编码器）
"""

d_k = 64
d_v = 64
d_embedding = 512

n_heads = 8
n_layers = 6

assert d_k * n_heads == d_embedding and d_v * n_heads == d_embedding

# [ENC-2-ABS] 绝对位置编码（正弦）生成表
def get_sin_enc_table(n_position, embedding_dim):
    sinusoid_table = np.zeros((n_position, embedding_dim))
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i / np.power(10000, 2 * (hid_j // 2) / embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)

# [ATTN-core] 缩放点积注意力（用于 Multi-Head 内部）
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V, attn_mask):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (d_k ** 0.5)
        scores.masked_fill_(attn_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn

# [ENC-3]/[DEC-1]/[DEC-2] Multi-Head Attention（含 QKV 映射与注意力矩阵）
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads)  # Multi-Attn 的 Q映射
        self.W_K = nn.Linear(d_embedding, d_k * n_heads)  # Multi-Attn 的 K映射
        self.W_V = nn.Linear(d_embedding, d_v * n_heads)  # Multi-Attn 的 V映射
        self.linear = nn.Linear(n_heads * d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)       # [ENC-5] 残差+LayerNorm / 解码同理
        self.attn = ScaledDotProductAttention()

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
        context, weights = self.attn(q_s, k_s, v_s, attn_mask)  # 注意力矩阵与加权
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        output = self.layer_norm(output + residual)  # 残差 + LN
        return output, weights

# PAD 掩码（自注意/交叉注意都会用到）
def get_attn_pad_mask(seq_q, seq_k):
    B, Lq = seq_q.size()
    _, Lk = seq_k.size()
    pad_attn_mask = seq_k.eq(0).unsqueeze(1).expand(B, Lq, Lk)  # True 表示要屏蔽
    return pad_attn_mask

# [ENC-4] 前馈网络（FFN + 激活）; 解码器层里也会用
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=d_embedding, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_embedding)  # [ENC-5] 残差+LayerNorm
    def forward(self, inputs):
        residual = inputs
        output = torch.relu(self.conv1(inputs.transpose(1, 2)))  # 激活函数
        output = self.conv2(output).transpose(1, 2)
        output = self.layer_norm(output + residual)              # 残差 + LN
        return output

# ---------------- Encoder ----------------
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  # [ENC-3] 自注意
        self.pos_ffn = PoswiseFeedForwardNet()     # [ENC-4] FFN
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn_weights = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn_weights

class Encoder(nn.Module):
    def __init__(self, corpus, max_len=4096):
        super().__init__()
        self.src_emb = nn.Embedding(corpus.src_vocab, d_embedding, padding_idx=0)
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sin_enc_table(max_len, d_embedding), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    def forward(self, enc_inputs):
        B, L = enc_inputs.size()
        pos_idx = torch.arange(L, device=enc_inputs.device).unsqueeze(0)  # 通常从0开始
        x = self.src_emb(enc_inputs) + self.pos_emb(pos_idx)
        mask = get_attn_pad_mask(enc_inputs, enc_inputs).to(enc_inputs.device)
        attn_ws = []
        for layer in self.layers:
            x, w = layer(x, mask)
            attn_ws.append(w)
        return x, attn_ws

# ---------------- Decoder ----------------
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()  # [DEC-1] Masked Self-Attn
        self.dec_enc_attn = MultiHeadAttention()   # [DEC-2] Cross-Attn
        self.pos_ffn = PoswiseFeedForwardNet()     # FFN + 残差+LN
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)  # [DEC-1]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)  # [DEC-2]
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

# 解码器用的“看未来”屏蔽：上三角为 True
def get_attn_subsequent_mask(seq):
    B, L = seq.size(0), seq.size(1)
    subsequent_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=seq.device), diagonal=1)
    return subsequent_mask.unsqueeze(0).expand(B, L, L)

class Decoder(nn.Module):
    def __init__(self, corpus, max_len=4096):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(corpus.tgt_vocab, d_embedding, padding_idx=0)  # 解码端 Embedding
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sin_enc_table(max_len, d_embedding), freeze=True)  # 用固定大表，防越界
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        B, L = dec_inputs.size()
        pos_indices = torch.arange(L, device=dec_inputs.device).unsqueeze(0)
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos_indices)
        pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)                   # PAD
        subsequent = get_attn_subsequent_mask(dec_inputs)                      # [DEC-1] Mask
        dec_self_attn_mask = (pad_mask.to(dec_inputs.device) | subsequent)     # 合并成最终 masked self-attn
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs).to(dec_inputs.device)  # 给 cross-attn 用
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                                             dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

# ---------------- Transformer 包装 ----------------
class Transformer(nn.Module):
    def __init__(self, corpus):
        super(Transformer, self).__init__()
        self.encoder = Encoder(corpus)
        self.decoder = Decoder(corpus)
        self.projection = nn.Linear(d_embedding, corpus.tgt_vocab, bias=False)  # 映射到词表
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # [batch, tgt_len, tgt_vocab]
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns

if __name__ == "__main__":
    print("设备选择：" + device)

    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)

    class Corpus:
        src_vocab, tgt_vocab = 1000, 1200
        src_len,  tgt_len  = 20, 22
    model = Transformer(Corpus()).to(device)
    src = torch.randint(1, Corpus.src_vocab, (2, Corpus.src_len)).to(device)
    tgt = torch.randint(1, Corpus.tgt_vocab, (2, Corpus.tgt_len)).to(device)
    print(model(src, tgt)[0].shape)  # 期望输出:[2, 22, 1200]



