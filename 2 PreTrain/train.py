from datasets import load_dataset

# 配置参数
k_workers = 32
data_dir = "./data"
file_pattern = f"{data_dir}/pretrain_*.txt"

dataset = load_dataset("text", data_files=file_pattern,
                       cache_dir="/root/lanyun-tmp/hf/cache",
                       sample_by="line",
                       num_proc=k_workers)
split_dataset = dataset["train"].train_test_split(train_size=0.01, seed=42)
train_dataset = split_dataset["train"]

import os
import torch
import matplotlib.pyplot as plt
from itertools import chain
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

output_path = "./output"
model_path = ("/root/lanyun-tmp/hf/Qwen2.5-0.5B")
config = AutoConfig.from_pretrained(model_path)
# 调整模型配置
config.num_attention_heads = 16
config.num_key_value_heads = 4
config.hidden_size = 1024
config.num_hidden_layers = 36
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def preprocess_dataset(examples):
    # 预处理数据集
    input_ids = tokenizer(examples["text"], return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
    return {"input_ids": input_ids}


train_dataset = train_dataset.map(
    preprocess_dataset,
    batched=True,
    batch_size=5000,
    remove_columns=train_dataset.column_names,
    num_proc=k_workers,
)

print(model.config)
print(f"该模型的总参数量为: {sum(p.numel() for p in model.parameters())}")

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    learning_rate=3e-3,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    save_steps=500,  # 保存中间模型
    save_total_limit=3,
    bf16=True,
    logging_steps=20,
    deepspeed="./ds_config.json",  # 关键参数
    remove_unused_columns=False
)

# 使用 Trainer 自动集成
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
)
trainer.train()
