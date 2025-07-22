import argparse
import os
import json
from pathlib import Path
from typing import List, Dict
from datasets import Dataset, concatenate_datasets
from dotenv import load_dotenv
from shutil import rmtree

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import PeftModel, LoraConfig, get_peft_model
from huggingface_hub import login
# import wandb

from steps import ExtractPaper, TaskCreation
from config import BASE_MODEL_PATH, LORA_MODEL_PATH, CONTINUAL_LEARNING_MODEL_PATH

load_dotenv()
# wandb.login()

# To save time on the AISA GPU Cluster, we downloaded the base model and saved it locally
# LOCAL_BASE_MODEL_PATH = r"/mnt/beegfs/home/st191428/base_model"

def load_tokenizer(local_model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(local_model_path: str, device: str):
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device_id = torch.cuda.current_device()
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            quantization_config=bnb_config,
            device_map={"": device_id},
            trust_remote_code=True,
        )
    else:
        raise RuntimeError("Quantized models must be loaded on a CUDA device.")

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    return model

def tokenize_function(examples, tokenizer, block_size):
    return tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True, max_length=block_size)

def group_texts(examples, block_size):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def load_and_prepare_dataset(file_path: Path, tokenizer, block_size):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or not all(isinstance(doc, dict) and "chunks" in doc for doc in data):
            print(f"Invalid JSON structure in {file_path.name} — skipping")
            return None, None
    except Exception as e:
        print(f"Error reading {file_path.name}: {e} — skipping")
        return None, None

    text_data = [chunk["text"] for doc in data for chunk in doc.get("chunks", []) if chunk.get("text", "").strip()]
    if not text_data or len(text_data) < 10:
        print(f"Skipping {file_path.name} — not enough valid text chunks.")
        return None, None

    raw_dataset = Dataset.from_list([{"text": t} for t in text_data])
    raw_dataset = raw_dataset.train_test_split(test_size=0.05)

    train_tokenized = raw_dataset["train"].map(lambda x: tokenize_function(x, tokenizer, block_size), batched=True, remove_columns=["text"])
    val_tokenized = raw_dataset["test"].map(lambda x: tokenize_function(x, tokenizer, block_size), batched=True, remove_columns=["text"])

    train_grouped = train_tokenized.map(lambda x: group_texts(x, block_size), batched=True)
    val_grouped = val_tokenized.map(lambda x: group_texts(x, block_size), batched=True)

    return train_grouped, val_grouped

def get_training_arguments(output_dir, batch_size, gradient_accumulation_steps, num_train_epochs, save_steps, logging_steps):
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        fp16=False,
        bf16=True,
        # report_to="wandb",
        eval_steps=500,
        do_eval=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        torch_compile=False,
        weight_decay=0.01,
    )

def train_on_task(
    model_path: str,
    task_input_dir: str,
    output_dir: str,
    block_size: int = 2048,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 3,
    save_steps: int = 1000,
    logging_steps: int = 100,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer(model_path)
    base_model = load_model(model_path, device)

    if CONTINUAL_LEARNING_MODEL_PATH:
        print(f"Loading LoRA model from {CONTINUAL_LEARNING_MODEL_PATH}")
        model = PeftModel.from_pretrained(base_model, CONTINUAL_LEARNING_MODEL_PATH).to(device)
    else:
        model = base_model

    task_files = sorted(Path(task_input_dir).glob("task_*.json"))
    print(f"Found {len(task_files)} task files in {task_input_dir}")

    for i, task_file in enumerate(task_files):
        print(f"\nProcessing Task {i+1}/{len(task_files)}: {task_file.name}")
        train_dataset, val_dataset = load_and_prepare_dataset(task_file, tokenizer, block_size)

        if train_dataset is None:
            print(f"Skipping {task_file.name} due to invalid or insufficient data.")
            continue

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        task_output_dir = Path(output_dir) / f"{task_file.stem}_checkpoint"
        task_output_dir.mkdir(parents=True, exist_ok=True)

        training_args = get_training_arguments(
            output_dir=str(task_output_dir),
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps,
            logging_steps=logging_steps
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Saving the checkpoints
        trainer.train()
        trainer.save_model(str(task_output_dir))
        tokenizer.save_pretrained(str(task_output_dir))

    # Save final version of the model
    # final_model_dir = Path(output_dir) / "final_model"
    # final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(CONTINUAL_LEARNING_MODEL_PATH))
    tokenizer.save_pretrained(str(CONTINUAL_LEARNING_MODEL_PATH))
    print(f"\n✅ Finished training on all tasks. Final model saved to: {CONTINUAL_LEARNING_MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=str, required=True, help='Path to PDF file')
    args = parser.parse_args()

    pdf_path = args.pdf

    extract_content = ExtractPaper()
    tasks = TaskCreation()

    # paper_content = []
    # extracted_paper_content = '/mnt/beegfs/home/st191428/extracted_paper_content.json'

    task_data = []
    task_id = 1
    current_task = []
    current_tokens = 0

    # Extracting and cleaning the text from the PDF
    paper_content = extract_content.process_pdf(pdf_path=pdf_path)

    # Creating tasks based on the paper data
    chunks = paper_content.get("chunks", [])
    full_text = "\n".join(chunk.get("text", "") for chunk in chunks)
    tokens = tasks.estimate_tokens(full_text)
    task_data.append({
        "content": paper_content,
        "tokens": tokens
    })
    print(f"Task with estimated {tokens} tokens")

    for paper in task_data:
        paper_tokens = paper["tokens"]
        if current_tokens + paper_tokens > tasks.MAX_TOKENS_PER_TASK and current_task:
            print(f"Writing task_{task_id:05d}.json with {len(current_task)} papers, total tokens: {current_tokens}")
            with open(f"/mnt/beegfs/home/st191428/tasks/task_{task_id:05d}.json", "w") as f:
                json.dump(current_task, f, indent=2)
            task_id += 1
            current_task = []
            current_tokens = 0

        current_task.append(paper["content"])
        current_tokens += paper_tokens

    # Final leftover task
    if current_task:
        print(f"Writing task_{task_id:05d}.json with {len(current_task)} papers, total tokens: {current_tokens}")
        with open(f"/mnt/beegfs/home/st191428/tasks/task_{task_id:05d}.json", "w") as f:
            json.dump(current_task, f, indent=2)

    print(f"Created {task_id}!")

    train_on_task(
        model_path = BASE_MODEL_PATH,
        task_input_dir = '/mnt/beegfs/home/st191428/tasks',
        output_dir = '/mnt/beegfs/home/st191428/llama3_medical_continual_learning')
            
if __name__ == '__main__':
    main()