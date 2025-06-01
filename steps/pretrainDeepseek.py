import os
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import json
from datasets import Dataset

class PretrainDeepseekLLM:
    def __init__(
        self,
        model_id: str = "deepseek-ai/deepseek-llm-r1",
        data_dir: str = "data",
        output_dir: str = "deepseek-liver-llm",
        block_size: int = 2048,
        batch_size: int = 2,
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
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)

        self.dataset = self._load_and_prepare_dataset()
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        self.training_args = self._get_training_arguments()

    def _load_and_prepare_dataset(self):
        data_files = list(self.data_dir.glob("*.json"))
        all_texts = []
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                for entry in json_data:
                    for chunk in entry['chunks']:
                        all_texts.append(chunk['text'])

        dataset = Dataset.from_dict({"text": all_texts})

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.block_size,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=["text"]
        )

        return tokenized_dataset

    def _get_training_arguments(self):
        return TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_train_epochs,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            learning_rate=2e-5,
            warmup_steps=500,
            weight_decay=0.01,
            fp16=True,
            report_to="none"
        )

    def train(self):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

        trainer.train()
        trainer.save_model(str(self.output_dir))
