import json
import os
import torch
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch import nn
from datasets import Dataset, Features, Sequence, Value
from pathlib import Path
from peft import PeftModel
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorWithPadding,
)
from trl import SFTTrainer
import wandb
import glob

# Load environment and login to wandb
os.environ["HF_HOME"] = "/path_to_GPU/home/Liver-LLM/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/path_to_GPU/home/Liver-LLM/hf_cache"
load_dotenv()
wandb.login()


class QA_Finetuning_MOA:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    def load_merged_qa(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    def process_qa(self, qa):
        question = qa.get("question", "").strip()
        answer = qa.get("answer", "").strip().lower()
        qa_type = qa.get("type", "MOA").strip().upper()
        
        if qa_type == "MOA":
            prompt = (
                f"### INSTRUCTION:\n"
                f"You are a helpful medical assistant trained on liver diseases.\n"
                f"Answer the following question concisely and accurately.\n"
                f"### INPUT:\n"
                f"QUESTION: {question}\n"
                f"### RESPONSE:\n"
                f"ANSWER: {answer}"
            )
            return {"text": prompt, "type": "MOA"}    

    def main(self):
        from config import LORA_MODEL_PATH, BASE_MODEL_PATH, QA_DATASET_PATH, QA_FINETUNED_MOA_MODEL_PATH

        # Load and process dataset
        qa_dataset = self.load_merged_qa(QA_DATASET_PATH)
        processed_data = []
        for qa in qa_dataset:
            result = self.process_qa(qa)
            if result is not None:
                processed_data.append(result)
        #processed_data = [self.process_qa(qa) for qa in qa_dataset]
        
        moa_data = [ex for ex in processed_data if ex['type'] == 'MOA']
        moa_dataset = Dataset.from_list(moa_data)

        features = Features({
            "input_ids": Sequence(Value("int64")),
            "attention_mask": Sequence(Value("int64")),
            "labels": Sequence(Value("int64"))
        })

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        # Tokenization
        def tokenize_moa(ex):
            output = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=256)
            output['labels'] = output['input_ids'].copy()
            print(output[0])
            return {
                "input_ids": output["input_ids"],
                "attention_mask": output["attention_mask"],
                "labels": output["labels"]
            }
        moa_tokenized = moa_dataset.map(tokenize_moa, remove_columns=["text", "type"])
        print(moa_tokenized[0])
        moa_tokenized = moa_tokenized.cast(features)
        moa_tokenized_split = moa_tokenized.train_test_split(test_size=0.1, seed=42)
        print("Before set_format keys (train):", moa_tokenized_split["train"].features)
        moa_tokenized_split["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        print("After set_format keys (train):", moa_tokenized_split["train"][0])

        #moa_tokenized_split["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        print("Before set_format keys (test):", moa_tokenized_split["test"].features)
        moa_tokenized_split["test"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        print("After set_format keys (test):", moa_tokenized_split["test"][0])
        
        print('Tokenised datasets loaded')
        if self.device == "cuda":
            print("CUDA available: Using 4-bit quantization with bitsandbytes.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_id = torch.cuda.current_device()
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                quantization_config=bnb_config,
                device_map={"": device_id},
                offload_folder="/path_to_GPU/home/Liver-LLM/offload",
                trust_remote_code=True,
            )
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model.config, "pretraining_tp"):
            model.config.pretraining_tp = 1
        #model.config.pretraining_tp = 1
        # Load base and LoRA models
        #base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, low_cpu_mem_usage=True)
        lora_model = PeftModel.from_pretrained(model, LORA_MODEL_PATH, local_files_only=True)
        lora_model.enable_input_require_grads()
        print('LoRA model loaded')

        # TrainingArguments
        
        moa_args = TrainingArguments(
            output_dir=os.path.join(QA_FINETUNED_MOA_MODEL_PATH, "moa_model"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            save_steps=1000,
            save_total_limit=3,
            num_train_epochs=3,
            learning_rate=2e-4,
            weight_decay = 0.01,
            warmup_ratio=0.03,
            bf16=True,
            gradient_checkpointing=True,
            #torch_compile=True,
            report_to="wandb"
        )

        wandb.init(project="liver_multitask", name="moa_finetuning")
        # MOA Trainer
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        moa_trainer = SFTTrainer(
            model=lora_model,
            args=moa_args,
            train_dataset=moa_tokenized_split['train'],
            eval_dataset=moa_tokenized_split['test'],
            data_collator=data_collator
        )
        print(
            torch.cuda.memory_summary(
                device=None,
                abbreviated=False
            )
        )
        # Clearing GPU memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        checkpoint_dir = os.path.join(QA_FINETUNED_MOA_MODEL_PATH, "moa_model")
        checkpoints = sorted(glob.glob(f"{checkpoint_dir}/checkpoint-*"), key=os.path.getmtime)

        if checkpoints:
            print(f"Resuming training from checkpoint: {checkpoints[-1]}")
            moa_trainer.train(resume_from_checkpoint=checkpoints[-1])
        else:
            moa_trainer.train()
        moa_trainer.save_model(QA_FINETUNED_MOA_MODEL_PATH)
        tokenizer.save_pretrained(QA_FINETUNED_MOA_MODEL_PATH)

        print("Finetuning complete!")
