import json
import os
from config import saved_liver_llm_model, saved_liver_llm_qa_model
import json
from pathlib import Path
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

class QA_Finetuning:
    def read_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def process_qa_entry(self, entry):
        processed_entry = {}
        processed_entry["question"] = entry.get("question", "").strip()
        if "options" in entry.keys():
            processed_entry["type"] = "MCQ"
            processed_entry["options"] = entry.get("options", [])
            processed_entry["answer"] = entry.get("cop", "").strip()
        else:
            processed_entry["type"] = "MOA"
            processed_entry["answer"] = entry.get("answer", "").strip()
        
        
        return processed_entry

    def merge_datasets(self, file_paths):
        merged_data = []
        for file_path in file_paths:
            data = self.read_json(file_path)
            for entry in data:
                processed_entry = self.process_qa_entry(entry)
                merged_data.append(processed_entry)
        return merged_data

    def save_as_jsonl(self, data, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')

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
            options = qa.get("options", {})
            # ensure consistent order: a, b, c, d
            options_str = "\n".join(
                [f"{chr(65 + idx)}) {options.get(letter, '').strip()}" 
                for idx, letter in enumerate(['a', 'b', 'c', 'd'])]
            )
            prompt = (
                f"<|question|>: {question}\n"
                f"<|options|>:\n{options_str}\n"
                f"<|answer|>: {answer}"
            )
        else:
            prompt = f"<|question|>: {question}\n<|answer|>: {answer}"
        return {"text": prompt}


    def main(self):
        # Paths to your two datasets
        dataset_paths = ['test_qa_mcq.json', 'test_qa_moa.json']
        merged_data = self.merge_datasets(dataset_paths)
        
        output_file = 'liver_qa_data.jsonl'
        self.save_as_jsonl(merged_data, output_file)
        
        print(f"Processed {len(merged_data)} QA pairs and saved to {output_file}")

        qa_dataset = self.load_merged_qa(output_file)

        processed_data = [self.process_qa(qa) for qa in qa_dataset]

        # Convert to Hugging Face dataset
        dataset = Dataset.from_list(processed_data)

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(saved_liver_llm_model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(saved_liver_llm_model)

        def tokenize_function(example):
            return tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=1024,
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Define Trainer
        training_args = TrainingArguments(
            output_dir=saved_liver_llm_qa_model,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            fp16=True,
            report_to="none"
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        # Start training
        trainer.train()

        # Save final model
        trainer.save_model(saved_liver_llm_qa_model)
        tokenizer.save_pretrained(saved_liver_llm_qa_model)

        print(f"✅ Finetuning complete! Model saved to {saved_liver_llm_qa_model}")


if __name__ == '__main__':
    qa =QA_Finetuning()
    qa.main()