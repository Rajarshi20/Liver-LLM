import json
import os
import torch
import numpy as np
from torch import nn
from datasets import Dataset
from pathlib import Path
from peft import PeftModel
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from trl import SFTTrainer
import wandb

# Set HF cache directories
os.environ["HF_HOME"] = "/path_to_GPU/home/Liver-LLM/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/path_to_GPU/home/Liver-LLM/hf_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load environment and login to wandb
load_dotenv()
wandb.login()

class MultiTaskLiverModel(nn.Module):
    def __init__(self, base_model, num_labels=5):
        super().__init__()
        self.base_model = base_model
        self.classifier = None  # Lazy init
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None, task_type="mcq", **kwargs):
        task_type = kwargs.get("task_type", "mcq")

        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )

        if task_type == "mcq":
            last_hidden_state = base_outputs.hidden_states[-1]
            pooled_output = last_hidden_state[:, 0, :]

            if self.classifier is None:
                dtype = pooled_output.dtype
                device = pooled_output.device
                hidden_size = pooled_output.shape[-1]
                self.classifier = nn.Linear(hidden_size, self.num_labels).to(device=device, dtype=dtype)

            logits = self.classifier(pooled_output)
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}

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
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_merged_qa(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    def process_qa(self, qa):
        question = qa.get("question", "").strip()
        answer = qa.get("answer", "").strip().lower()
        qa_type = qa.get("type", "MCQ").strip().upper()

        if qa_type == "MCQ":
            explanation = qa.get("explanation", "")
            options = qa.get("options", {})
            valid_labels = sorted([k.lower() for k in options if k.lower() in ["a", "b", "c", "d", "e"]])

            if answer not in valid_labels:
                print(f"Skipping invalid MCQ answer: '{answer}' not in options: {valid_labels}")
                return None

            options_str = "\n".join([f"{label}. {options.get(label, '').strip()}" for label in valid_labels])
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

    def main(self):
        from config import LORA_MODEL_PATH, BASE_MODEL_PATH, QA_DATASET_PATH, QA_FINETUNED_MCQ_MODEL_PATH

        qa_dataset = self.load_merged_qa(QA_DATASET_PATH)
        processed_data = [self.process_qa(qa) for qa in qa_dataset if self.process_qa(qa)]
        mcq_data = [ex for ex in processed_data if ex['type'] == 'MCQ']
        mcq_dataset = Dataset.from_list(mcq_data)

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        def tokenize_mcq(ex):
            out = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=256)
            out["labels"] = ex["label"]
            return out

        mcq_tokenized = mcq_dataset.map(tokenize_mcq, remove_columns=["text", "label", "type"])
        mcq_tokenized_split = mcq_tokenized.train_test_split(test_size=0.1)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map={"": torch.cuda.current_device()},
            offload_folder="/path_to_GPU/home/Liver-LLM/offload",
            trust_remote_code=True,
        )

        lora_model = PeftModel.from_pretrained(model, LORA_MODEL_PATH, local_files_only=True)
        lora_model.enable_input_require_grads()
        lora_model.config.use_cache = False

        multitask_model = MultiTaskLiverModel(lora_model)
        multitask_model.base_model.gradient_checkpointing_enable()

        training_args = TrainingArguments(
            output_dir=QA_FINETUNED_MCQ_MODEL_PATH,
            run_name="mcq_finetuning",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            num_train_epochs=3,
            save_steps=1000,
            save_total_limit=3,
            fp16=True,
            bf16=False,
            logging_steps=100,
            report_to="wandb"
        )

        def compute_mcq_metrics(eval_pred):
            preds = np.argmax(eval_pred.predictions, axis=1)
            labels = eval_pred.label_ids
            return {"accuracy": (preds == labels).astype(np.float32).mean()}

        wandb.init(project="liver_multitask", name="mcq_finetuning")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

        trainer = Trainer(
            model=multitask_model,
            args=training_args,
            train_dataset=mcq_tokenized_split['train'],
            eval_dataset=mcq_tokenized_split['test'],
            compute_metrics=compute_mcq_metrics,
            data_collator=data_collator
        )

        trainer.train()

        merged_model = multitask_model.base_model.merge_and_unload()
        merged_model.save_pretrained(QA_FINETUNED_MCQ_MODEL_PATH + "_merged")
        tokenizer.save_pretrained(QA_FINETUNED_MCQ_MODEL_PATH + "_merged")
        print("Finetuning complete!")
