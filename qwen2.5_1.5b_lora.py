# encoding=utf-8

import os
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from torch.optim import AdamW

# 配置GPU设备（单GPU环境，多GPU可调整为["0,1"]）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 关闭Tokenizer并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 核心参数配置（根据实际环境修改路径与超参）
CONFIG = {
    "base_model_path": "Qwen2.5-1.5B-Instruct",  # 基础模型路径
    "train_data_path": "bank_intent_data\train.jsonl",  # 训练集路径
    "test_data_path": "bank_intent_data\test.jsonl",  # 测试集路径
    "lora_save_path": "bank_lora_model",  # LoRA权重保存路径（统一存放所有文件）
    "labels": ["fraud_risk", "refund", "balance", "card_lost", "other"],  # 意图标签列表
    "max_seq_len": 64,  # 文本最大截断长度（适配语料长度分布）
    "batch_size": 1,  # 批量大小（规避Qwen pad_token兼容问题）
    "epochs": 1,  # 训练轮次（平衡拟合效果与过拟合风险）
    "lr": 2e-5  # LoRA学习率（基于Qwen-1.5B最优经验值）
}

# 标签映射（文本标签→数字ID，模型输入要求）
label2id = {label: idx for idx, label in enumerate(CONFIG["labels"])}
id2label = {idx: label for idx, label in enumerate(CONFIG["labels"])}


def load_and_process_data():
    # 加载语料（单个JSONL文件默认拆分为"train" split）
    train_ds = load_dataset("json", data_files=CONFIG["train_data_path"], split="train")
    test_ds = load_dataset("json", data_files=CONFIG["test_data_path"], split="train")

    # 数据清洗：过滤标签不在预设列表中的无效样本
    def clean_data(examples):
        valid_mask = [label in CONFIG["labels"] for label in examples["label"]]
        return {
            "text": [text for text, mask in zip(examples["text"], valid_mask) if mask],
            "label": [label2id[label] for label, mask in zip(examples["label"], valid_mask) if mask]
        }

    train_ds = train_ds.map(clean_data, batched=True, remove_columns=train_ds.column_names)
    test_ds = test_ds.map(clean_data, batched=True, remove_columns=test_ds.column_names)

    # 初始化Tokenizer（Qwen模型需手动设置pad_token）
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model_path"],
        trust_remote_code=True,  # 加载Qwen自定义模型必要参数
        padding_side="right"  # 右侧padding，避免影响模型注意力计算
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 文本分词：统一长度+固定padding（避免后续动态padding兼容问题）
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,  # 截断超max_seq_len的文本
            max_length=CONFIG["max_seq_len"],  # 统一输入长度
            padding="max_length"  # 固定长度padding，确保batch内样本维度一致
        )

    tokenized_train = train_ds.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_test = test_ds.map(tokenize_function, batched=True, remove_columns=["text"])

    # 标签列重命名：模型要求标签列名为"labels"
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    # 打印数据基本信息（用于调试与验证）
    print(f"数据预处理完成：训练集样本数={len(tokenized_train)}，测试集样本数={len(tokenized_test)}")
    return tokenized_train, tokenized_test, tokenizer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 从logits中取概率最大的类别作为预测结果
    predictions = np.argmax(logits, axis=-1)
    # 计算Macro-F1（对所有类别平等加权，适配类别分布不均衡场景）
    macro_f1 = f1_score(
        y_true=labels,
        y_pred=predictions,
        average="macro",
        labels=list(label2id.values())  # 仅计算预设标签的F1，排除无效标签干扰
    )
    return {"macro_f1": round(macro_f1, 4)}


def main():
    # 确保目标文件夹存在（避免路径错误）
    os.makedirs(CONFIG["lora_save_path"], exist_ok=True)

    # 加载预处理后的数据
    tokenized_train, tokenized_test, tokenizer = load_and_process_data()

    # 基础模型（无LoRA）评估：获取 baseline F1分数
    print("\n=== 基础模型（无LoRA）评估 ===")
    # 加载基础模型（序列分类任务）
    base_model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model_path"],
        trust_remote_code=True,
        num_labels=len(CONFIG["labels"]),  # 分类任务类别数=标签数
        id2label=id2label,
        label2id=label2id,
        device_map="auto",  # 自动分配设备（优先GPU，无GPU则用CPU）
        dtype=torch.float32  # 用FP32规避FP16梯度兼容问题
    )
    # 初始化分类头权重（Qwen预训练模型无分类头，需手动初始化）
    if hasattr(base_model, "score"):
        torch.nn.init.normal_(base_model.score.weight, mean=0.0, std=0.02)

    # 基础模型评估配置（日志放入bank_lora_model/base_eval）
    base_trainer = Trainer(
        model=base_model,
        args=TrainingArguments(
            output_dir=f"{CONFIG['lora_save_path']}/base_eval",  # 统一路径
            per_device_eval_batch_size=CONFIG["batch_size"],
            do_train=False,  # 仅评估，不训练
            do_eval=True,  # 执行评估
            remove_unused_columns=False  # 保留所有必要列，避免数据丢失
        ),
        eval_dataset=tokenized_test,  # 测试集作为评估数据集
        compute_metrics=compute_metrics  # 评估指标函数
    )
    # 执行评估并记录baseline F1
    base_eval_result = base_trainer.evaluate()
    base_f1 = base_eval_result["eval_macro_f1"]
    print(f"基础模型（无LoRA）Macro-F1: {base_f1}")

    # 释放基础模型显存（避免占用后续训练资源）
    del base_model, base_trainer
    torch.cuda.empty_cache()

    # LoRA微调训练
    print("\n=== LoRA微调训练 ===")
    # 重新加载基础模型（用于注入LoRA）
    lora_model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model_path"],
        trust_remote_code=True,
        num_labels=len(CONFIG["labels"]),
        id2label=id2label,
        label2id=label2id,
        device_map="auto",
        dtype=torch.float32
    )
    # 初始化分类头权重
    if hasattr(lora_model, "score"):
        torch.nn.init.normal_(lora_model.score.weight, mean=0.0, std=0.02)

    # 配置LoRA参数（基于Qwen-1.5B最优经验值）
    lora_config = LoraConfig(
        r=8,  # LoRA秩：控制低秩矩阵维度，平衡效果与参数量
        lora_alpha=16,  # 缩放因子：alpha = 2*r，增强梯度更新幅度
        target_modules=["q_proj", "v_proj"],  # 目标层：仅训练注意力Q/V投影层
        lora_dropout=0.05,  # Dropout：降低过拟合风险
        bias="none",  # 不训练偏置参数：减少计算量
        task_type="SEQ_CLS"  # 任务类型：序列分类（Sequence Classification）
    )
    # 注入LoRA到基础模型
    lora_model = get_peft_model(lora_model, lora_config)
    # 打印可训练参数占比（验证LoRA参数效率）
    print("LoRA参数占比：")
    lora_model.print_trainable_parameters()

    # 配置训练参数（日志放入bank_lora_model/lora_training）
    training_args = TrainingArguments(
        output_dir=f"{CONFIG['lora_save_path']}/lora_training",  # 统一路径
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        num_train_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["lr"],
        logging_steps=50,  # 日志打印间隔：每50步打印一次训练状态
        eval_strategy="epoch",  # 评估策略：每轮训练后评估
        save_strategy="epoch",  # 保存策略：每轮训练后保存模型
        save_total_limit=1,  # 保存模型数量：仅保留效果最好的1个
        load_best_model_at_end=True,  # 训练结束后加载最优模型
        fp16=False,  # 关闭FP16：规避梯度缩放兼容问题
        report_to="none",  # 关闭第三方日志报告（简化流程）
        remove_unused_columns=False
    )

    # 初始化优化器（AdamW：适配LoRA微调的常用优化器）
    optimizer = AdamW(
        lora_model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=0.01  # 权重衰减：降低过拟合风险
    )

    # 初始化Trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_train,  # 训练数据集
        eval_dataset=tokenized_test,  # 评估数据集
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)  # 传入自定义优化器
    )

    # 执行LoRA微调
    trainer.train()

    # 保存LoRA权重（放入bank_lora_model根目录）
    lora_model.save_pretrained(CONFIG["lora_save_path"])
    print(f"\nLoRA权重已保存至：{CONFIG['lora_save_path']}")

    # 释放训练显存
    del lora_model, trainer, optimizer
    torch.cuda.empty_cache()

    # LoRA微调后模型评估
    print("\n=== LoRA微调后模型评估 ===")
    # 加载基础模型 + LoRA权重
    final_model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model_path"],
        trust_remote_code=True,
        num_labels=len(CONFIG["labels"]),
        id2label=id2label,
        label2id=label2id,
        device_map="auto",
        dtype=torch.float32
    )
    # 合并LoRA权重到基础模型（模拟部署场景）
    final_model = PeftModel.from_pretrained(final_model, CONFIG["lora_save_path"])
    final_model.merge_and_unload()

    # 评估微调后模型（日志放入bank_lora_model/final_eval）
    final_trainer = Trainer(
        model=final_model,
        args=TrainingArguments(
            output_dir=f"{CONFIG['lora_save_path']}/final_eval",  # 统一路径
            per_device_eval_batch_size=CONFIG["batch_size"],
            do_train=False,
            do_eval=True,
            remove_unused_columns=False
        ),
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )
    final_eval_result = final_trainer.evaluate()
    final_f1 = final_eval_result["eval_macro_f1"]
    print(f"LoRA微调后模型Macro-F1: {final_f1}")

    # 输出微调前后效果对比
    print("\n=== LoRA微调前后效果对比 ===")
    print(f"基础模型（无LoRA）Macro-F1: {base_f1}")
    print(f"LoRA微调后模型Macro-F1: {final_f1}")
    print(f"Macro-F1提升幅度: {round(final_f1 - base_f1, 4)}")


if __name__ == "__main__":
    main()



