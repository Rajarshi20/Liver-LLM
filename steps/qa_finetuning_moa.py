import json
import os
import torch
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch import nn
from datasets import Dataset
from pathlib import Path
from peft import PeftModel
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from trl import SFTTrainer
import wandb

# Load environment and login to wandb
os.environ["HF_HOME"] = "/hkfs/work/workspace/scratch/st_st191428-LiverLLM/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/hkfs/work/workspace/scratch/st_st191428-LiverLLM/hf_cache"
load_dotenv()
wandb.login()


class QA_Finetuning_MOA:

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
        from config import LORA_MODEL_PATH, BASE_MODEL_PATH, QA_DATASET_PATH, QA_FINETUNED_MODEL_PATH

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

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        # Tokenization
        def tokenize_moa(ex):
            output = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=256, return_tensors = "pt")
            output['labels'] = output['input_ids'].clone()
            return output

        moa_tokenized = moa_dataset.map(tokenize_moa, remove_columns=["text", "type"])
        moa_tokenized_split = moa_tokenized.train_test_split(test_size=0.1)
        
        print('Tokenised datasets loaded')

        # Load base and LoRA models
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, low_cpu_mem_usage=True)
        lora_model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH, local_files_only=True)
        lora_model.enable_input_require_grads()
        print('LoRA model loaded')
        
        # Pass LoRA-wrapped model into multitask wrapper
        #multitask_model = MultiTaskLiverModel(lora_model)


        # TrainingArguments
        
        moa_args = TrainingArguments(
            output_dir=os.path.join(QA_FINETUNED_MODEL_PATH, "moa_model"),
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
            torch_compile=True,
            report_to="wandb"
        )

        
        # MOA Trainer
        data_collator = default_data_collator
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

        # Training
        
        wandb.init(project="liver_multitask", name="moa_finetuning")
        moa_trainer.train()
        moa_trainer.save_model(os.path.join(QA_FINETUNED_MODEL_PATH, "moa_model"))
        tokenizer.save_pretrained(os.path.join(QA_FINETUNED_MODEL_PATH, "moa_model"))

        print("Finetuning complete!")
