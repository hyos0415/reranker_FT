import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import wandb
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="BAAI/bge-reranker-v2-m3")

@dataclass
class DataArguments:
    train_data_path: str = field(default="data/hard_train_triplets.jsonl")
    val_data_path: str = field(default="data/hard_val_triplets.jsonl")
    max_length: int = field(default=512)

def train():
    model_args = ModelArguments()
    data_args = DataArguments()
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    # 2. BitsAndBytes Config (4-bit quantization for VRAM efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["classifier"] # BGE-reranker (XLMRoberta) uses 'classifier'
    )
    
    # 3. Load Model with PEFT
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        num_labels=1,
        dtype=torch.bfloat16, # Changed from torch_dtype
        device_map="auto",
        trust_remote_code=True
    )
    
    # Ensure classification head is in float32 for training
    if hasattr(model, "classifier"):
        model.classifier.to(torch.float32)
    
    model = prepare_model_for_kbit_training(model)
    
    # BGE-reranker-v2-m3 target modules (XLMRoberta based)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "key", "value"], 
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["classifier"] # Explicitly save the head in full precision
    )
    
    model = get_peft_model(model, lora_config)
    
    # Re-verify dtypes for trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype not in [torch.float32, torch.bfloat16, torch.float16]:
            param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()
    
    # 4. Dataset Preparation
    # JSONL format: {"query": "...", "pos": ["..."], "neg": ["..."]}
    dataset = load_dataset('json', data_files={
        "train": data_args.train_data_path,
        "validation": data_args.val_data_path
    })

    def tokenize_function(examples):
        # Flattening for re-ranker training (pairs of query and passage)
        # However, for simplicity in LoRA, we can use the default pair handling
        # BGE reranker expects [query, passage] pairs
        queries = []
        passages = []
        labels = []
        
        for i in range(len(examples['query'])):
            # Positive pair
            queries.append(examples['query'][i])
            passages.append(examples['pos'][i][0])
            labels.append(1.0)
            
            # Negative pairs (multiple hard negatives)
            for neg in examples['neg'][i]:
                queries.append(examples['query'][i])
                passages.append(neg)
                labels.append(0.0)
            
        tokenized = tokenizer(
            queries,
            passages,
            truncation=True,
            max_length=data_args.max_length,
            padding="max_length"
        )
        tokenized["labels"] = labels
        return tokenized

    train_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    val_dataset = dataset["validation"].map(tokenize_function, batched=True, remove_columns=dataset["validation"].column_names)
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="models/reranker-peft-v1",
        per_device_train_batch_size=32, # Proven stable speed for this environment
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        bf16=True,
        push_to_hub=False,
        report_to="wandb",
        run_name="korean-reranker-peft-v1",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer, # Changed from tokenizer for newer transformers
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 7. Start Training
    wandb.init(project="korean-reranker-peft", name="reranker-peft-v1")
    print("Starting Training...")
    trainer.train()
    print("Training Completed.")

if __name__ == "__main__":
    train()
