import os
import json
from pathlib import Path
from typing import List, Dict
from datasets import Dataset, concatenate_datasets
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
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import wandb

os.environ["HF_HOME"] = "/hkfs/work/workspace/scratch/st_st191428-LiverLLM/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/hkfs/work/workspace/scratch/st_st191428-LiverLLM/hf_cache"
load_dotenv()
wandb.login()

class NewPretrainLLM:
    LLAMA_API = os.getenv('LLAMA')

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B",
        data_dir: str = "adv_cleaned_tasks",
        output_dir: str = "llama3_medical_finetuned",
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
        os.makedirs(output_dir, exist_ok=True)
        self.block_size = block_size
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.save_steps = save_steps
        self.logging_steps = logging_steps

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self._load_model()
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        self.training_args = self._get_training_arguments()

    def _load_model(self):
        if self.device == "cuda":
            print("CUDA available: Using 4-bit quantization with bitsandbytes.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_id = torch.cuda.current_device()
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
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
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        print("Model loaded")
        return model

    def _load_and_prepare_dataset(self, file_path: Path) -> Dataset:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            text_data = [chunk["text"] for doc in data for chunk in doc.get("chunks", []) if chunk.get("text", "").strip()]
        raw_dataset = Dataset.from_list([{"text": t} for t in text_data])
        raw_dataset = raw_dataset.train_test_split(test_size=0.05)

        train_tokenized = raw_dataset["train"].map(self._tokenize_function, batched=True, remove_columns=["text"])
        train_grouped = train_tokenized.map(self._group_texts, batched=True)

        val_tokenized = raw_dataset["test"].map(self._tokenize_function, batched=True, remove_columns=["text"])
        val_grouped = val_tokenized.map(self._group_texts, batched=True)

        return train_grouped, val_grouped

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True, max_length=self.block_size)

    def _group_texts(self, examples):
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
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=2,
            num_train_epochs=self.num_train_epochs,
            logging_steps=self.logging_steps,
            fp16=False,
            bf16=True,
            report_to="wandb",
            evaluation_strategy="steps",
            eval_steps=500,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            torch_compile=False,
            weight_decay=0.01,
        )

    def continual_loop(self):
        login(self.LLAMA_API)
        replay_buffer = []
        json_files = sorted(self.data_dir.glob("*.json"))

        wandb.init(
            project="llama3_medical",
            entity="annie-ch-university-of-stuttgart",
            job_type="training", 
            anonymous="allow",
            name="continual_adaptation_run"
        )
        for i, file_path in enumerate(json_files):
            print(f"\nTask {i+1}/{len(json_files)}: {file_path.name}")
            current_train, current_val = self._load_and_prepare_dataset(file_path)

            # Optionally repeat current data 3x for stronger learning
            weighted_current = concatenate_datasets([current_train] * 3)
            combined_dataset = [weighted_current] + [Dataset.load_from_disk(p) for p in replay_buffer]
            final_train_dataset = concatenate_datasets(combined_dataset).shuffle(seed=42)

            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=final_train_dataset,
                eval_dataset=current_val,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
            )

            trainer.train()
            wandb.finish()
            # Save current processed dataset to disk
            buffer_path = self.output_dir / f"task_{i}_buffer"
            current_train.save_to_disk(buffer_path)
            replay_buffer.append(buffer_path)

        self.model.save_pretrained(str(self.output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
