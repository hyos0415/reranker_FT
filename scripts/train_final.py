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
    EarlyStoppingCallback,
    HfArgumentParser
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
    train_data_path: str = field(default="data/final_train_triplets.jsonl")
    val_data_path: str = field(default="data/hard_val_triplets.jsonl")
    max_length: int = field(default=512)

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    # 2. BitsAndBytes Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["classifier"]
    )
    
    # 3. Load Model with PEFT
    # Check if we are loading from an existing adapter or a base model
    is_adapter = os.path.exists(os.path.join(model_args.model_name_or_path, "adapter_config.json"))
    
    if is_adapter:
        print(f"Loading existing adapter from {model_args.model_name_or_path}...")
        # Load base model first (BGE-M3 Reranker is usually XLMRoberta based)
        # We need the base model name. For this project, it's BAAI/bge-reranker-v2-m3
        base_model_name = "BAAI/bge-reranker-v2-m3" 
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            num_labels=1,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path, is_trainable=True)
    else:
        print(f"Loading base model and creating new LoRA: {model_args.model_name_or_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=bnb_config,
            num_labels=1,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        if hasattr(model, "classifier"):
            model.classifier.to(torch.float32)
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "key", "value"], 
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            modules_to_save=["classifier"]
        )
        model = get_peft_model(model, lora_config)
    
    # Re-verify dtypes
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype not in [torch.float32, torch.bfloat16, torch.float16]:
            param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()
    
    # 4. Dataset Preparation
    dataset = load_dataset('json', data_files={
        "train": data_args.train_data_path,
        "validation": data_args.val_data_path
    })

    def tokenize_function(examples):
        queries = []
        passages = []
        labels = []
        for i in range(len(examples['query'])):
            queries.append(examples['query'][i])
            passages.append(examples['pos'][i][0])
            labels.append(1.0)
            for neg in examples['neg'][i]:
                queries.append(examples['query'][i])
                passages.append(neg)
                labels.append(0.0)
        tokenized = tokenizer(
            queries, passages,
            truncation=True,
            max_length=data_args.max_length,
            padding="max_length"
        )
        tokenized["labels"] = labels
        return tokenized

    train_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    val_dataset = dataset["validation"].map(tokenize_function, batched=True, remove_columns=dataset["validation"].column_names)
    
    # 5. Final Training Configuration
    if training_args.output_dir == "tmp_trainer":
        training_args.output_dir = "models/reranker-peft-final"
    
    training_args.bf16 = True
    training_args.report_to = ["wandb"]
    training_args.run_name = "korean-reranker-final-stage"
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "loss"
    training_args.greater_is_better = False
    
    # Evaluation & Saving Strategy (Optimized for small dataset)
    training_args.eval_strategy = "epoch"
    training_args.save_strategy = "epoch"
    training_args.save_total_limit = 2
    
    training_args.per_device_train_batch_size = training_args.per_device_train_batch_size or 32
    training_args.learning_rate = training_args.learning_rate or 2e-4
    training_args.num_train_epochs = training_args.num_train_epochs or 1
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 7. Start Final Training
    wandb.init(project="korean-reranker-peft", name=training_args.run_name)
    print("Starting Final Stage Training (Merged Dataset)...")
    trainer.train()
    print("Final Training Completed.")

if __name__ == "__main__":
    train()
