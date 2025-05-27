from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

class PretrainLLM:
    # Load tokenizer and model (change this to actual LLaMA if using that)
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Tokenize text
    block_size = 2048  # depends on model and GPU capacity

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], return_special_tokens_mask=True)

    # Group into blocks
    def group_texts(self, examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // self.block_size) * self.block_size
        result = {
            k: [t[i : i+self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
        
    def main(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # or torch.bfloat16 if supported
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"  # Automatically assigns layers to available devices
        )

        # Load your cleaned and chunked JSONL file
        dataset = load_dataset("json", data_files={"train": "data/liver_chunks.jsonl"}, split="train")

        tokenized_dataset = dataset.map(self.tokenize_function, batched=True, remove_columns=["text"])

        lm_dataset = tokenized_dataset.map(self.group_texts, batched=True)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./llama-medical-pretrain",
            overwrite_output_dir=True,
            per_device_train_batch_size=2,  # Adjust based on GPU
            gradient_accumulation_steps=8,
            evaluation_strategy="no",
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=2,
            num_train_epochs=1,
            logging_steps=100,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU allows
            report_to="none",
            remove_unused_columns=False
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        # Start training
        trainer.train()

        # Save model
        trainer.save_model("./llama-medical-final")