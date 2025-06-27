import json
import os
import re
from config import LORA_MODEL_PATH, BASE_MODEL_PATH, QA_DATASET_PATH, QA_FINETUNED_MODEL_PATH
from pathlib import Path
from tqdm import tqdm
from peft import PeftModel
from trl import SFTTrainer, setup_chat_format
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

wandb.login()
class QA_Finetuning:
    
    def load_merged_qa(self, path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def process_qa(self, qa):
        question = qa.get("question", "").strip()
        answer = qa.get("answer", "").strip()
        qa_type = qa.get("type", "MOA").strip().upper()
        
        if qa_type == "MCQ":
            explanation = qa.get("explanation", "")
            options = qa.get("options", {})
            # ensure consistent order: a, b, c, d
            options_str = "\n".join(
                [f"{chr(65 + idx)}) {options.get(letter, '')}" 
                for idx, letter in enumerate(['a', 'b', 'c', 'd'])]
            )
            prompt = (
                f"### INSTRUCTION:\n"
                f"You are a helpful medical assistant who has been trained on liver diseases domain."
                f"Based on your training and domain expertise, answer the following question. Be as specific and accurate as possible.\n"
                f"### INPUT:\n"
                f"<|question|> : {question}\n"
                f"<|options|>:\n{options_str}\n"
                f"### CONTEXT:\n"
                f"<|explanation|> : {explanation}\n"
                f"### RESPONSE:\n"
                f"<|answer|> : {answer}"
            )
        else:
            prompt = (
                f"### INSTRUCTION:\n"
                f"You are a helpful medical assistant who has been trained on liver diseases domain."
                f"Based on your training and domain expertise, answer the following question. Be as specific and accurate as possible.\n"
                f"### INPUT:\n"
                f"<|question|>: {question}\n"
                f"### RESPONSE:\n"
                f"\n<|answer|>: {answer}"
            )
        return {"text": prompt}


    def main(self):

        qa_dataset = self.load_merged_qa(QA_DATASET_PATH)

        processed_data = [self.process_qa(qa) for qa in qa_dataset]

        # Convert to Hugging Face dataset
        dataset = Dataset.from_list(processed_data)

        # Load tokenizer and model
        # tokenizer = AutoTokenizer.from_pretrained(saved_liver_llm_model, use_fast=True)
        # model = AutoModelForCausalLM.from_pretrained(saved_liver_llm_model)

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
        print("This is the LORA Path :",LORA_MODEL_PATH)
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH,local_files_only=True)
        model.enable_input_require_grads()

        def tokenize_function(example):
            return tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=1024,
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_dataset_split = tokenized_dataset.train_test_split(test_size=0.1)

        # Define Trainer
        training_args = TrainingArguments(
            output_dir=QA_FINETUNED_MODEL_PATH,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            optim="adamw_torch",
            evaluation_strategy="steps",
            eval_steps=500,
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            torch_compile=True,
            report_to="wandb"
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # wandb
        wandb.init(
            project="llama3_medical_qa_finetuned",
            entity="annie-ch-university-of-stuttgart",
            job_type="training", 
            anonymous="allow",
            name="qa_run"
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset_split['train'],
            eval_dataset=tokenized_dataset_split['test'],
            max_seq_length=512,
            dataset_text_field="text",
            data_collator=data_collator
        )

        # Start training
        trainer.train()

        # Save final model
        trainer.save_model(QA_FINETUNED_MODEL_PATH)
        tokenizer.save_pretrained(QA_FINETUNED_MODEL_PATH)

        print(f"Finetuning complete! Model saved to {QA_FINETUNED_MODEL_PATH}")
