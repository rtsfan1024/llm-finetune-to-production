# encoding=utf-8

import os, json, argparse, math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# === 1) 依赖：导入 mini_transformer ===
import mini_transformer as mt   # 确保同目录下有 mini_transformer.py

# 固定随机种子（可重复）
torch.manual_seed(0)
if getattr(mt, "device", "cpu") == "cuda":
    torch.cuda.manual_seed_all(0)

# === 2) 数据与分词 ===
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

def load_parallel_pairs(path):
    """读取行对行平行语料：奇数行中文，偶数行英文；已保证干净。"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert len(lines) % 2 == 0, "文件行数应为偶数（中英交替）"
    pairs = []
    for i in range(0, len(lines), 2):
        cn = lines[i]
        en = lines[i+1]
        pairs.append((cn, en))
    return pairs

def tokenize_zh(s):
    return list(s)  # 字符级

def tokenize_en(s):
    return s.split()  # 词级（空格）

def build_vocab(seqs, min_freq=1):
    from collections import Counter
    cnt = Counter()
    for toks in seqs:
        cnt.update(toks)
    itos = [PAD, BOS, EOS, UNK]
    for w, f in cnt.items():
        if f >= min_freq and w not in itos:
            itos.append(w)
    stoi = {w:i for i,w in enumerate(itos)}
    return stoi, itos

def encode(tokens, stoi):
    return [stoi.get(t, stoi[UNK]) for t in tokens]

def pad_batch(batch_ids, pad_id=0):
    L = max(len(x) for x in batch_ids)
    return [x + [pad_id]*(L-len(x)) for x in batch_ids]

# === 3) 数据集与 DataLoader ===
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, src_stoi, tgt_stoi):
        self.src_stoi = src_stoi
        self.tgt_stoi = tgt_stoi
        self.data = []
        for zh, en in pairs:
            src_tok = tokenize_zh(zh)
            tgt_tok = tokenize_en(en)
            src_ids = encode(src_tok, src_stoi)
            # Decoder 输入以 BOS 开头；标签以 EOS 结尾
            dec_in  = encode([BOS] + tgt_tok, tgt_stoi)
            tgt_out = encode(tgt_tok + [EOS], tgt_stoi)
            self.data.append((src_ids, dec_in, tgt_out))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate_fn(batch):
    srcs, dins, tgts = zip(*batch)
    pad_id_src = 0  # 把 <pad> 放在 id=0
    pad_id_tgt = 0
    srcs = pad_batch(list(srcs), pad_id_src)
    dins = pad_batch(list(dins), pad_id_tgt)
    tgts = pad_batch(list(tgts), pad_id_tgt)
    return (torch.tensor(srcs, dtype=torch.long, device=mt.device),
            torch.tensor(dins, dtype=torch.long, device=mt.device),
            torch.tensor(tgts, dtype=torch.long, device=mt.device))

# === 4) 训练 ===
def train_model(corpus_file="mini_translation_pairs.txt",
                save_dir="./ckpt",
                epochs=3, batch_size=32, lr=3e-4):

    os.makedirs(save_dir, exist_ok=True)

    # 读取数据对
    pairs = load_parallel_pairs(corpus_file)

    # 构建词表
    src_tokens = [tokenize_zh(zh) for zh,_ in pairs]
    tgt_tokens = [tokenize_en(en) for _,en in pairs]

    src_stoi, src_itos = build_vocab(src_tokens)
    tgt_stoi, tgt_itos = build_vocab([[BOS]+t+[EOS] for t in tgt_tokens])

    # 确保 <pad> 的 id 为 0（遮罩代码假定 PAD=0）
    assert src_stoi[PAD] == 0 and tgt_stoi[PAD] == 0, "PAD 必须是 0"

    # 数据集与 DataLoader
    ds = TranslationDataset(pairs, src_stoi, tgt_stoi)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 统计最大长度，作为位置编码上限（留冗余以免越界）
    max_src_len = max(len(x[0]) for x in ds.data)
    max_tgt_len = max(len(x[1]) for x in ds.data)  # dec_in 长度
    src_len_cap = max_src_len + 16
    tgt_len_cap = max_tgt_len + 16

    # 构造一个“Corpus”对象（供 mini_transformer 初始化）
    Corpus = type("Corpus", (), dict(
        src_vocab=len(src_itos),
        tgt_vocab=len(tgt_itos),
        src_len=src_len_cap,
        tgt_len=tgt_len_cap,
    ))

    model = mt.Transformer(Corpus()).to(mt.device)

    crit = nn.CrossEntropyLoss(ignore_index=tgt_stoi[PAD])
    opt  = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs+1):
        total_loss, n_tok = 0.0, 0
        for src, dec_in, tgt_out in dl:
            opt.zero_grad()
            logits, *_ = model(src, dec_in)  # [B, T, V]
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            opt.step()
            # 统计
            valid_tokens = (tgt_out != tgt_stoi[PAD]).sum().item()
            total_loss += loss.item() * max(valid_tokens, 1)
            n_tok += max(valid_tokens, 1)
        ppl = math.exp(total_loss / max(n_tok, 1))
        print(f"[Epoch {epoch}] loss/token={total_loss/n_tok:.4f} | ppl={ppl:.2f}")

    # 保存：模型参数 + 词表 + 长度上限
    torch.save(model.state_dict(), os.path.join(save_dir, "mini_transformer_ckpt.pt"))
    meta = {
        "src_itos": src_itos,
        "tgt_itos": tgt_itos,
        "src_stoi": src_stoi,
        "tgt_stoi": tgt_stoi,
        "src_len": src_len_cap,
        "tgt_len": tgt_len_cap,
    }
    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("==> 训练完成，已保存到 ./ckpt/mini_transformer_ckpt.pt 与 ./ckpt/meta.json")

# === 5) 推断（贪心解码） ===
def greedy_decode(model, src_ids, tgt_stoi, max_len=80):
    model.eval()
    with torch.no_grad():
        src = torch.tensor([src_ids], device=mt.device)
        ys  = torch.tensor([[tgt_stoi[BOS]]], device=mt.device)
        for _ in range(max_len):
            logits, *_ = model(src, ys)
            next_id = int(logits[0, -1].argmax(-1))
            ys = torch.cat([ys, torch.tensor([[next_id]], device=mt.device)], dim=1)
            if next_id == tgt_stoi[EOS]:
                break
    return ys[0].tolist()[1:]  # 去掉 BOS

def load_model(save_dir="./ckpt"):
    with open(os.path.join(save_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    Corpus = type("Corpus", (), dict(
        src_vocab=len(meta["src_itos"]),
        tgt_vocab=len(meta["tgt_itos"]),
        src_len=meta["src_len"],
        tgt_len=meta["tgt_len"],
    ))
    model = mt.Transformer(Corpus()).to(mt.device)
    sd = torch.load(os.path.join(save_dir, "mini_transformer_ckpt.pt"), map_location=mt.device)
    model.load_state_dict(sd)
    return model, meta

def translate_one(zh_text, save_dir="./ckpt"):
    model, meta = load_model(save_dir)
    src_stoi, src_itos = meta["src_stoi"], meta["src_itos"]
    tgt_stoi, tgt_itos = meta["tgt_stoi"], meta["tgt_itos"]

    src_ids = [src_stoi.get(t, src_stoi[UNK]) for t in tokenize_zh(zh_text)]
    # 简单长度保护：若超过训练时的 src_len 上限，则截断
    if len(src_ids) > meta["src_len"]:
        src_ids = src_ids[:meta["src_len"]]

    out_ids = greedy_decode(model, src_ids, tgt_stoi, max_len=meta["tgt_len"])
    # 去掉 EOS 之后的部分
    out_tokens = []
    for idx in out_ids:
        if idx == tgt_stoi[EOS]:
            break
        out_tokens.append(meta["tgt_itos"][idx])
    return " ".join(out_tokens)

# === 6) 命令行入口 ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="训练并保存模型")
    ap.add_argument("--corpus", type=str, default="mini_translation_pairs.txt", help="平行语料文件")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save_dir", type=str, default="./ckpt")
    ap.add_argument("--translate", type=str, default=None, help="输入一条中文做翻译推断")
    args = ap.parse_args()

    if args.train:
        train_model(corpus_file=args.corpus,
                    save_dir=args.save_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr)

    if args.translate is not None:
        en = translate_one(args.translate, save_dir=args.save_dir)
        print("翻译结果:", en)

if __name__ == "__main__":
    main()



