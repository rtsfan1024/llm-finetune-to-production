# encoding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os

from mini_decoder_only import DecoderOnlyLM, device

class CorpusTokenizer:
    def __init__(self, corpus_path, pad_id=0, unk_id=1, eos_id=2):
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.eos_id = eos_id  # <sep>的ID
        self.vocab = self._build_vocab(corpus_path)
        self.id_to_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        print(f"从语料构建词汇表：共{self.vocab_size}个token（<pad>={pad_id}, <unk>={unk_id}, <sep>={eos_id}）")

    def _build_vocab(self, corpus_path):
        vocab = {
            "<pad>": self.pad_id,
            "<unk>": self.unk_id,
            "<sep>": self.eos_id
        }
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().replace(" ", "")
                if not line:
                    continue
                for char in line:
                    if char not in vocab:
                        vocab[char] = len(vocab)
        return vocab

    def encode(self, text, max_len):
        chars = text.replace(" ", "").strip()
        ids = [self.vocab.get(c, self.unk_id) for c in chars]
        ids.append(self.eos_id)
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids += [self.pad_id] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        text = ""
        for idx in ids:
            if idx == self.pad_id or idx == self.eos_id:
                break
            text += self.id_to_vocab.get(idx, "<unk>")
        return text

class TranslationCorpusDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)
        print(f"加载语料：{corpus_path}，共{len(self.data)}条样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        return self.tokenizer.encode(text, self.max_len)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Decoder-Only LM")
    parser.add_argument("--train", action="store_true", help="Train mode")
    parser.add_argument("--corpus", type=str, default="mini_translation_pairs.txt", help="Corpus path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./ckpt_gpt", help="Model save directory")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for generation")
    parser.add_argument("--max_len", type=int, default=64, help="Max sequence length")
    return parser.parse_args()

def train_model(args, tokenizer):
    dataset = TranslationCorpusDataset(
        corpus_path=args.corpus,
        tokenizer=tokenizer,
        max_len=args.max_len
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True if device == "cuda" else False
    )

    model = DecoderOnlyLM(
        vocab_size=tokenizer.vocab_size
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "mini_decoder_only_model.pth")

    model.train()
    print(f"\n开始训练（设备：{device}）")
    print(f"超参：epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            inputs = batch[:, :-1].to(device)
            labels = batch[:, 1:].to(device)

            logits, _, _ = model(inputs)
            logits_flat = logits.reshape(-1, tokenizer.vocab_size)
            labels_flat = labels.reshape(-1)

            loss = criterion(logits_flat, labels_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            if (step + 1) % 10 == 0:
                avg_loss = total_loss / ((step + 1) * args.batch_size)
                print(f"Epoch [{epoch + 1}/{args.epochs}], Step [{step + 1}/{len(dataloader)}], Loss: {avg_loss:.4f}")

        epoch_avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch + 1}/{args.epochs}] 结束，平均损失：{epoch_avg_loss:.4f}")

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': tokenizer.vocab
    }, save_path)
    print(f"模型已保存至：{save_path}")

def generate_text(args):
    save_path = os.path.join(args.save_dir, "mini_decoder_only_model.pth")
    if not os.path.exists(save_path):
        print(f"错误：未找到模型文件 {save_path}，请先训练模型")
        return

    checkpoint = torch.load(save_path, map_location=device)

    # 重建tokenizer
    tokenizer = CorpusTokenizer(args.corpus)
    tokenizer.vocab = checkpoint['vocab']
    tokenizer.id_to_vocab = {v: k for k, v in tokenizer.vocab.items()}
    tokenizer.vocab_size = len(tokenizer.vocab)

    # 修复：模型初始化参数
    model = DecoderOnlyLM(
        vocab_size=tokenizer.vocab_size
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 编码prompt
    prompt_ids = tokenizer.encode(args.prompt, max_len=len(args.prompt) + 1)
    prompt_ids = prompt_ids.unsqueeze(0).to(device)

    # 生成文本
    generated_ids = model.generate(
        idx=prompt_ids,
        max_new_tokens=50,
        eos_id=tokenizer.eos_id,
        temperature=0.8
    )

    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"\n生成结果: {generated_text}")

if __name__ == "__main__":
    args = parse_args()
    if args.train:
        tokenizer = CorpusTokenizer(args.corpus)
        train_model(args, tokenizer)
    elif args.prompt:
        generate_text(args)
    else:
        print("请指定训练模式（--train）或生成模式（--prompt）")



