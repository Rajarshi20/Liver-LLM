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
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer
import wandb

# Load environment and login to wandb
os.environ["HF_HOME"] = "/hkfs/work/workspace/scratch/st_st191428-LiverLLM/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/hkfs/work/workspace/scratch/st_st191428-LiverLLM/hf_cache"
load_dotenv()
wandb.login()

# Custom classification and generation model
class MultiTaskLiverModel(nn.Module):
    def __init__(self, base_model, num_labels=5):
        super().__init__()
        self.base_model = base_model  # Use already-loaded model (e.g., LoRA-wrapped)
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, task_type="mcq", **kwargs):
        task_type = kwargs.get("task_type", "mcq")
        base_outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True  # ? required for next line
        )
        """
        with torch.no_grad():
            pooled_output = base_outputs.hidden_states[-1][:, 0, :]
        pooled_output = pooled_output.detach()"""
        
        if task_type == "mcq":
            # Get the last hidden state (from the last layer)
            last_hidden_state = base_outputs.hidden_states[-1]  # shape: (batch, seq_len, hidden)
            pooled_output = last_hidden_state[:, 0, :]           # use [CLS]-style token position
            logits = self.classifier(pooled_output)
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        
        else:
            return self.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
    
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(**kwargs)
    
    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()


class QA_Finetuning_MCQ:

    def load_merged_qa(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    def process_qa(self, qa):
        question = qa.get("question", "").strip()
        answer = qa.get("answer", "").strip().lower()
        qa_type = qa.get("type", "MOA").strip().upper()
    
        if qa_type == "MCQ":
            explanation = qa.get("explanation", "")
            options = qa.get("options", {})
    
            # Dynamically determine valid option keys (sorted and lowercase)
            valid_labels = sorted([key.lower() for key in options.keys() if key.lower() in ["a", "b", "c", "d", "e"]])
            
            # Validate answer
            if answer not in valid_labels:
                print(f"Skipping invalid MCQ answer: '{answer}' not in options: {valid_labels}")
                return None
    
            # Build options string
            options_str = "\n".join([f"{label}. {options.get(label, '').strip()}" for label in valid_labels])
    
            # Prompt
            prompt = (
                f"### INSTRUCTION:\n"
                f"You are a helpful medical assistant trained on liver diseases.\n"
                f"Study the following multiple-choice question and select the correct option.\n"
                f"### INPUT:\n"
                f"QUESTION: {question}\n"
                f"OPTIONS:\n{options_str}\n"
            )
            if explanation:
                prompt += f"Explanation: {explanation}\n"
            prompt += f"Answer: {answer}"
    
            return {
                "text": prompt,
                "label": valid_labels.index(answer),
                "type": "MCQ"
            }
        """
        else:
            # For MOA / open-ended questions
            prompt = (
                f"### INSTRUCTION:\n"
                f"You are a helpful medical assistant trained on liver diseases.\n"
                f"Study and respond accurately to the question below.\n"
                f"### INPUT:\n"
                f"QUESTION: {question}\n"
                f"### RESPONSE:\n"
                f"ANSWER: {answer}"
            )
            return {"text": prompt, "type": "MOA"}
        """
    def main(self):
        from config import LORA_MODEL_PATH, BASE_MODEL_PATH, QA_DATASET_PATH, QA_FINETUNED_MCQ_MODEL_PATH

        # Load and process dataset
        qa_dataset = self.load_merged_qa(QA_DATASET_PATH)
        processed_data = []
        for qa in qa_dataset:
            result = self.process_qa(qa)
            if result is not None:
                processed_data.append(result)
        #processed_data = [self.process_qa(qa) for qa in qa_dataset]
        mcq_data = [ex for ex in processed_data if ex['type'] == 'MCQ']
        #moa_data = [ex for ex in processed_data if ex['type'] == 'MOA']

        mcq_dataset = Dataset.from_list(mcq_data)
        #moa_dataset = Dataset.from_list(moa_data)

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        # Tokenization
        def tokenize_mcq(ex):
            out = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=50)
            out["labels"] = ex["label"]
            return out

        #def tokenize_moa(ex):
        #    return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=1024)

        mcq_tokenized = mcq_dataset.map(tokenize_mcq, remove_columns=["text", "label", "type"])
        #moa_tokenized = moa_dataset.map(tokenize_moa, remove_columns=["text", "type"])
        
        #mcq_tokenized_split = mcq_tokenized.train_test_split(test_size=0.1)
        mcq_tokenized_split = mcq_tokenized.train_test_split(test_size=100)
        
        #moa_tokenized_split = moa_tokenized.train_test_split(test_size=0.1)
        
        print('Tokenised datasets loaded')

        # Load base and LoRA models
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, low_cpu_mem_usage=True)
        lora_model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH, local_files_only=True)
        lora_model.config.use_cache = False
        #lora_model.enable_input_require_grads()
        print('LoRA model loaded')
        
        # Pass LoRA-wrapped model into multitask wrapper
        multitask_model = MultiTaskLiverModel(lora_model)
        multitask_model.base_model.gradient_checkpointing_enable()
        multitask_model.base_model.model.gradient_checkpointing_enable()
        
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        multitask_model.base_model.config.use_cache = False


        # TrainingArguments
        mcq_args = TrainingArguments(
            output_dir=QA_FINETUNED_MCQ_MODEL_PATH,
            run_name="mcq_finetuning",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            num_train_epochs=3,
            #evaluation_strategy="epoch",
            save_steps=1000,
            save_total_limit=3,
            bf16=False,
            fp16=True,
            logging_steps=100,
            report_to="wandb"
        )
        """
        moa_args = TrainingArguments(
            output_dir=os.path.join(QA_FINETUNED_MODEL_PATH, "moa_model"),
            per_device_train_batch_size=1,
            #gradient_accumulation_steps=2,
            save_steps=1000,
            save_total_limit=3,
            num_train_epochs=3,
            learning_rate=2e-4,
            bf16=True,
            gradient_checkpointing=True,
            torch_compile=True,
            report_to="wandb"
        )"""

        # Metric for MCQ
        
        def compute_mcq_metrics(eval_pred):
            preds = np.argmax(eval_pred.predictions, axis=1)
            labels = eval_pred.label_ids
            return {"accuracy": (preds == labels).astype(np.float32).mean()}

        mcq_trainer = Trainer(
            model=multitask_model,
            args=mcq_args,
            train_dataset=mcq_tokenized_split['train'],
            eval_dataset=mcq_tokenized_split['test'],
            compute_metrics=compute_mcq_metrics
        )

        # MOA Trainer
        """
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        moa_trainer = SFTTrainer(
            model=lora_model,
            args=moa_args,
            train_dataset=moa_tokenized_split['train'],
            eval_dataset=moa_tokenized_split['test'],
            data_collator=data_collator
        )"""
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
        wandb.init(project="liver_multitask", name="mcq_finetuning")
        mcq_trainer.train()
        mcq_trainer.save_model(QA_FINETUNED_MCQ_MODEL_PATH)
        tokenizer.save_pretrained(QA_FINETUNED_MCQ_MODEL_PATH)
        """
        wandb.init(project="liver_multitask", name="moa_finetuning")
        moa_trainer.train()
        moa_trainer.save_model(os.path.join(QA_FINETUNED_MODEL_PATH, "moa_model"))
        tokenizer.save_pretrained(os.path.join(QA_FINETUNED_MODEL_PATH, "moa_model"))
        """

        print("Finetuning complete!")
