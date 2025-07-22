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
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import wandb

from config import BASE_MODEL_PATH

os.environ["HF_HOME"] = "/path_to_GPU_/workspace/scratch/LiverLLM/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/path_to_GPU/work/workspace/scratch/LiverLLM/hf_cache"
load_dotenv()
wandb.login()

class NewPretrainLLM:
    LLAMA_API = os.getenv('LLAMA')

    def __init__(
        self,
        model_id: str = BASE_MODEL_PATH,
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
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        print("Model loaded")
        return model

    def _load_and_prepare_dataset(self, file_path: Path) -> Dataset:
        try:
          with open(file_path, "r", encoding="utf-8") as f:
              data = json.load(f)
  
          # ? Check if it's a list of dicts with "chunks" field
          if not isinstance(data, list) or not all(isinstance(doc, dict) and "chunks" in doc for doc in data):
              print(f"Invalid JSON structure in {file_path.name} skipping")
              return None, None

        except json.JSONDecodeError:
            print(f"Failed to decode JSON in {file_path.name} skipping")
            return None, None
        except Exception as e:
            print(f"Error reading {file_path.name}: {e} skipping")
            return None, None
            
        text_data = [chunk["text"] for doc in data for chunk in doc.get("chunks", []) if chunk.get("text", "").strip()]
            
        if not text_data:
            print(f"Skipping {file_path.name} no valid text chunks found.")
            return None, None
        
        if len(text_data) < 10:
            print(f"Skipping {file_path.name}: not enough valid text chunks ({len(text_data)}).")
            return None, None

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
            #evaluation_strategy="steps",
            eval_steps=500,
            do_eval= True,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            torch_compile=False,
            weight_decay=0.01,
        )

    def continual_loop(self):
        login(self.LLAMA_API)
        replay_buffer = []
        json_files = sorted(self.data_dir.glob("*.json"))
        
        #completed_tasks = {int(p.stem.split("_")[1]) for p in self.output_dir.glob("task_*_buffer")}
        #MAX_REPLAY_BUFFERS = 10
        #all_buffers = sorted(self.output_dir.glob("task_*_buffer"), key=lambda p: int(p.stem.split("_")[1]))
        #replay_buffer = all_buffers[-MAX_REPLAY_BUFFERS:]
        
        resume_checkpoint = self.output_dir / "checkpoint-1"
        start_task_index = 3908
        

        wandb.init(
            project="llama3_medical",
            entity="annie-ch-university-of-stuttgart",
            job_type="training", 
            anonymous="allow",
            name="continual_adaptation_run"
        )
        for i, file_path in enumerate(json_files):
          if i < start_task_index:
            continue #skiiping task files that are completed
          #if i in completed_tasks:
          #  print(f"Skipping already processed task {i}")
          #  continue
    
          print(f"\nTask {i+1}/{len(json_files)}: {file_path.name}")
          
          current_train, current_val = self._load_and_prepare_dataset(file_path)
          

          if current_train is None:
              print(f"Skipping task {i} due to empty or invalid data.")
              continue
              
          if len(current_train) < 2:
            print(f"Skipping task {i} too few samples after tokenization.")
            continue
            

          # Optionally repeat current data 3x for stronger learning
          final_train_dataset = current_train.shuffle(seed=42)

          trainer = Trainer(
              model=self.model,
              args=self.training_args,
              train_dataset=final_train_dataset,
              eval_dataset=current_val,
              tokenizer=self.tokenizer,
              data_collator=self.data_collator,
          )
          
          current_task_name = file_path.name

          # Save checkpoint BEFORE processing task_06900.json
          #if current_task_name == "task_06970.json":
              #ckpt_path = self.output_dir / "checkpoint_before_task_06970"
              #print(f"Saving checkpoint before processing {current_task_name}")
              #self.model.save_pretrained(str(ckpt_path))
              #self.tokenizer.save_pretrained(str(ckpt_path))

          if i == start_task_index:
            trainer.train(resume_from_checkpoint=str(resume_checkpoint))
          else:
            trainer.train()
          wandb.finish()
          
          # Save current processed dataset to disk
          #buffer_path = self.output_dir / f"task_{i}_buffer"
          #current_train.save_to_disk(buffer_path)
          #replay_buffer.append(buffer_path)
          
          if i == 7105:
            ckpt_path = self.output_dir / f"checkpoint_{i}"
            print(f" Saving checkpoint at task {i}")
            trainer.save_model(str(ckpt_path))
            self.tokenizer.save_pretrained(str(ckpt_path))
          
          

        self.model.save_pretrained(str(self.output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
