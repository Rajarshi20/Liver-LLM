import os
import json
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

load_dotenv()

class PretrainLLM:
    LLAMA_API = os.getenv('LLAMA')

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        data_dir: str = "data",
        output_dir: str = "llama4-medical-finetuned",
        block_size: int = 8192,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 8,
        num_train_epochs: int = 1,
        save_steps: int = 1000,
        logging_steps: int = 100,
    ):
        self.model_id = model_id
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.block_size = block_size
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.save_steps = save_steps
        self.logging_steps = logging_steps

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = self._load_model()
        self.dataset = self._load_and_prepare_dataset()
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        self.training_args = self._get_training_arguments()

    def _load_model(self):
        if self.device == "cuda":
            print("✅ CUDA available: Using 4-bit quantization with bitsandbytes.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            print("⚠️ CUDA not available: Loading model in float32 on CPU.")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=None,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            model.to(self.device)

        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        return model

    def _load_and_prepare_dataset(self) -> Dataset:
        all_files = list(self.data_dir.glob("*.json"))
        datasets_list = []

        for file_path in all_files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for doc in data:
                    for chunk in doc.get("chunks", []):
                        text = chunk.get("text", "")
                        if text.strip():
                            datasets_list.append({"text": text})

        dataset = Dataset.from_list(datasets_list)
        tokenized_dataset = dataset.map(
            self._tokenize_function, batched=True, remove_columns=["text"]
        )
        lm_dataset = tokenized_dataset.map(self._group_texts, batched=True)

        return lm_dataset

    def _tokenize_function(self, examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        return self.tokenizer(examples["text"], return_special_tokens_mask=True)

    def _group_texts(self, examples: Dict[str, List[int]]) -> Dict[str, List[List[int]]]:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // self.block_size) * self.block_size
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def _get_training_arguments(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            evaluation_strategy="no",
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=2,
            num_train_epochs=self.num_train_epochs,
            logging_steps=self.logging_steps,
            fp16=self.device == "cuda",
            report_to="none",
            remove_unused_columns=False,
        )

    def train(self):
        login(self.LLAMA_API)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        trainer.train()
        trainer.save_model(str(self.output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
