import json
import torch
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import QA_FINETUNED_MCQ_MODEL_PATH, QA_FINETUNED_MCQ_VALIDATION_DATASET

# Classifier wrapper
class MultiTaskLiverModel(nn.Module):
    def __init__(self, base_model, num_labels=5):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, task_type="mcq", **kwargs):
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        if task_type == "mcq":
            last_hidden = base_outputs.hidden_states[-1]  # (batch_size, seq_len, hidden)
            cls_token = last_hidden[:, 0, :]               # take [CLS]-like token at position 0
            logits = self.classifier(cls_token)

            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)

            return {"loss": loss, "logits": logits}
        else:
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

class FineTunedModelEvaluationMCQ:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model_and_tokenizer(self, model_path):
        print(f"Loading model from: {model_path}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map={"": torch.cuda.current_device()} if self.device == "cuda" else "auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        model = MultiTaskLiverModel(base_model)
        target_dtype = next(model.base_model.parameters()).dtype
        model.classifier = model.classifier.to(dtype=target_dtype, device=self.device)
        model.to(self.device)
        model.eval()

        return model, tokenizer

    def load_dataset(self, dataset_path):
        print(f"Loading dataset from: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = [json.loads(line) for line in f if line.strip()]

        dataset = []
        for qa in raw_data:
            if qa.get("type", "").upper() != "MCQ":
                continue

            question = qa.get("question", "").strip()
            options = qa.get("options", {})
            answer = qa.get("answer", "").strip().lower()

            valid_options = sorted([k.lower() for k in options if k.lower() in ["a", "b", "c", "d", "e"]])
            if answer not in valid_options:
                continue

            label_index = valid_options.index(answer)
            lower_to_actual = {k.lower(): k for k in options if k.lower() in valid_options}
            options_str = "\n".join([f"{k}. {options[lower_to_actual[k]]}" for k in valid_options])

            prompt = (
                f"### INSTRUCTION:\n"
                f"You are a helpful medical assistant trained on liver diseases.\n"
                f"Study the following multiple-choice question and select the correct option.\n"
                f"### INPUT:\n"
                f"QUESTION: {question}\n"
                f"OPTIONS:\n{options_str}\n"
                f"Answer:"
            )

            dataset.append({
                "prompt": prompt,
                "label_index": label_index
            })

        return dataset

    def main(self):
        model, tokenizer = self.load_model_and_tokenizer(QA_FINETUNED_MCQ_MODEL_PATH + "_merged")
        data = self.load_dataset(QA_FINETUNED_MCQ_VALIDATION_DATASET)

        total, correct = 0, 0

        for i, item in enumerate(data, 1):
            try:
                prompt = item["prompt"]
                label = item["label_index"]

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=256,
                ).to(self.device)

                with torch.no_grad():
                    outputs = model(**inputs, task_type="mcq")
                    logits = outputs["logits"]
                    pred = torch.argmax(logits, dim=1).item()

                print(f"[Q{i}] Predicted: {pred} | Actual: {label}")
                if pred == label:
                    correct += 1
                total += 1

            except Exception as e:
                print(f"[Skip Q{i}] Error: {str(e)}")

        print("\nEvaluation complete.")
        print(f"Total: {total} | Correct: {correct}")
        print(f"MCQ Accuracy: {correct / total:.4f}" if total else "MCQ Accuracy: 0.0000")
