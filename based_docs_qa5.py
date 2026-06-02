# encoding=utf-8

import os
import json
import torch
import transformers
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

# 加在三元组训练数据
def load_triple_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"三元组数据文件不存在：{data_path}，请先准备数据")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 验证数据格式
    required_keys = ["instruction", "input", "output"]
    for i, item in enumerate(data):
        if not all(key in item for key in required_keys):
            raise ValueError(f"第{i}条数据格式错误，需包含{required_keys}")

    print(f"成功加载三元组数据，共{len(data)}条样本")
    return data

def format_train_text(example):
    """
    将三元组转换为模型训练格式（指令微调常用格式）
    格式："### 指令：{instruction}\n### 输入：{input}\n### 输出：{output}"
    """
    return f"""### 指令：{example['instruction']}
### 输入：{example['input']}
### 输出：{example['output']}"""

def find_all_linear_names(model):
    """找到模型中所有线性层（用于LoRA适配）"""
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # 排除lm_head，避免影响输出
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_base_model(model_path):
    """加载基础模型（DeepSeek-7B-base），启用4位量化节省内存"""
    # 4位量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",  # 自动分配设备（优先GPU，无GPU则用CPU）
        trust_remote_code=True
    )

    # 准备模型用于量化训练
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

def setup_lora(model, r=8, lora_alpha=32, lora_dropout=0.05):
    """配置LoRA参数并应用到模型"""
    # 找到所有线性层
    modules = find_all_linear_names(model)

    # LoRA配置
    lora_config = LoraConfig(
        r=r,  # 秩，控制LoRA矩阵维度（越小参数越少）
        lora_alpha=lora_alpha,
        target_modules=modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 应用LoRA到模型
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数比例（通常<1%）
    return model

# 训练lora模型
def train_lora(model, tokenizer, dataset, output_dir="./lora_results"):
    # 处理数据集：tokenize训练文本
    def tokenize_function(examples):
        texts = [format_train_text(example) for example in examples]
        return tokenizer(
            texts,
            truncation=True,
            max_length=1024,
            padding="max_length",
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names  # 移除原始文本列，只保留tokenized数据
    )

    # 训练参数配置（轻量级微调，适合CPU/GPU）
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,  # 批次大小（CPU设1，GPU可设2-4）
        gradient_accumulation_steps=4,  # 梯度累积，模拟大批次
        learning_rate=2e-4,  # LoRA常用学习率
        num_train_epochs=3,  # 训练轮次（小数据集3-5轮足够）
        logging_steps=10,  # 每10步打印一次日志
        save_steps=50,  # 每50步保存一次模型
        warmup_ratio=0.1,  # 学习率预热比例
        optim="paged_adamw_8bit",  # 8位优化器，节省内存
        report_to="none",  # 不使用wandb等日志工具
        push_to_hub=False  # 不推送到hub
    )

    # 开始训练
    model.train()
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )
    trainer.train()

    # 保存LoRA权重（仅保存增量参数，约几十MB）
    model.save_pretrained(os.path.join(output_dir, "lora_weights"))
    print(f"LoRA微调完成，权重保存至：{os.path.join(output_dir, 'lora_weights')}")
    return output_dir

# 主函数：LoRA微调全流程
if __name__ == "__main__":
    # 配置路径
    BASE_MODEL_PATH = "/Users/zhaoshuai/models/deepseek-llm-7b-base"  # 基础模型路径（同第四阶段）
    TRIPLE_DATA_PATH = "./triple_data.json"  # 三元组数据文件路径
    OUTPUT_DIR = "./lora_results"  # 微调结果保存路径

    # 加载三元组数据
    triple_data = load_triple_data(TRIPLE_DATA_PATH)
    dataset = Dataset.from_list(triple_data)  # 转换为HuggingFace Dataset格式

    # 加载基础模型
    model, tokenizer = load_base_model(BASE_MODEL_PATH)

    # 配置LoRA
    model = setup_lora(model)

    # 开始微调
    train_lora(model, tokenizer, dataset, OUTPUT_DIR)

    # 提示后续步骤
    print("\n下一步：在第四阶段问答代码中加载LoRA权重，测试微调效果")
    print("加载方式：使用peft的PeftModel.from_pretrained(base_model, lora_weights_path)")



