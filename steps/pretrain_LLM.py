import os
import json
from pathlib import Path
from typing import List, Dict
from datasets import Dataset
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
from config import pretrain_model_id, pretrain_data_dir, pretrain_output_dir

load_dotenv()

class PretrainLLM:
    LLAMA_API = os.getenv('LLAMA')

    def __init__(
        self,
        block_size: int = 2048,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        num_train_epochs: int = 1,
        save_steps: int = 1000,
        logging_steps: int = 100,
    ):
        os.makedirs(pretrain_output_dir, exist_ok=True)
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
        device_id = torch.cuda.current_device()
        model = AutoModelForCausalLM.from_pretrained(
            pretrain_model_id,
            quantization_config=bnb_config,
            device_map={"": device_id},
            trust_remote_code=True,
        )
      else:
        raise RuntimeError("❌ Quantized models must be loaded on a CUDA device.")

      # Enable gradient checkpointing for memory efficiency
      model.gradient_checkpointing_enable()
      model.config.use_cache = False
      model.config.pretraining_tp = 1
  
      # Apply LoRA
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


    def _load_and_prepare_dataset(self) -> Dataset:
    
      all_files = list(pretrain_data_dir.glob("*.json"))
      datasets_list = []

      for file_path in all_files:
          with open(file_path, "r", encoding="utf-8") as f:
              try:
                  data = json.load(f)
                  for doc in data:
                      for chunk in doc.get("chunks", []):
                          text = chunk.get("text", "")
                          if text.strip():
                              datasets_list.append({"text": text})
              except json.JSONDecodeError:
                  print(f"❌ Skipping corrupt file: {file_path}")
                  continue
  
      # Save memory by mapping with memory-efficient loading
      raw_dataset = Dataset.from_list(datasets_list)
  
      # Optional: Save and reload with mmap to avoid keeping in memory
      raw_dataset.save_to_disk("raw_dataset.arrow")
      raw_dataset = Dataset.load_from_disk("raw_dataset.arrow")
  
      # Tokenize with minimal memory usage
      tokenized_dataset = raw_dataset.map(
          self._tokenize_function,
          batched=True,
          remove_columns=["text"],
          num_proc=1  # reduce from 4 to 1 to lower RAM usage
      )
  
      # Group texts for training blocks
      lm_dataset = tokenized_dataset.map(
          self._group_texts,
          batched=True,
          num_proc=1  # reduce to avoid overloading CPU RAM
      )
      print("Load and prepared dataset executed")
  
      return lm_dataset


    def _tokenize_function(self, examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        print("Tokenization done")
        return self.tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True, max_length=4096)

    def _group_texts(self, examples: Dict[str, List[int]]) -> Dict[str, List[List[int]]]:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // self.block_size) * self.block_size
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        print("Inside group text completed")
        return result

    def _get_training_arguments(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(pretrain_output_dir),
            overwrite_output_dir=True,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            #evaluation_strategy="no",
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=2,
            num_train_epochs=self.num_train_epochs,
            logging_steps=self.logging_steps,
            fp16=False,
            report_to="none",
            remove_unused_columns=False,
            gradient_checkpointing=True,
            bf16=True,  # If supported on H100
            torch_compile=False,  # Optional for performance

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
        trainer.save_model(str(pretrain_output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(pretrain_output_dir / "final_model"))
