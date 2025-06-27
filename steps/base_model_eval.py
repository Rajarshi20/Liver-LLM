import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from peft import PeftModel
from datasets import load_dataset
import evaluate
import nltk
import os
import csv
#nltk.download("punkt_tab")

nltk.data.path.append("/hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver LLM/liverenv/lib/nltk_data/tokenizers/")

from nltk.tokenize import word_tokenize
import json


from config import BASE_MODEL_PATH, QA_DATASET_PATH

class BaseModelEvaluation:
    def __init__(self):
        self.MAX_NEW_TOKENS = 128
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model_and_tokenizer(self, base_path, lora_path):
        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_path)
        model.eval().to(self.DEVICE)
        pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0 if self.DEVICE == "cuda" else -1)
        return tokenizer, model, pipeline

    def load_general_qa(self, path):
        general_qa = []
        with open(path, "r", encoding="utf-8") as f:
          for line in f:
            line = line.strip()
            if not line:
              continue
            obj = json.loads(line)
            if obj.get("type") == "MOA":
              general_qa.append({"question": obj["question"], "answer": obj["answer"]})
        return general_qa
    
    def load_mcq_qa(self, path):
        mcq_qa = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") == "MCQ":
                    mcq_qa.append({
                        "question": obj["question"],
                        "options": obj["options"],
                        "correct_option": obj["answer"]
                    })
        return mcq_qa

    def generate_general_answers(self, pipeline, dataset):
        predictions = []
        for item in dataset:
            prompt = f"Question: {item['question']}\nAnswer:"
            output = pipeline(prompt, max_new_tokens=self.MAX_NEW_TOKENS, pad_token_id=pipeline.tokenizer.eos_token_id)[0]['generated_text']
            answer = output.split("Answer:")[-1].strip()
            predictions.append(answer)
        return predictions

    def generate_mcq_answers(self, pipeline, dataset):
        predictions, references = [], []
        for item in dataset:
            prompt = f"Question: {item['question']}\n"
            for key, value in item['options'].items():
                prompt += f"{key}. {value}\n"
            prompt += "Answer (a/b/c/d):"
            output = pipeline(prompt, max_new_tokens=10, pad_token_id=pipeline.tokenizer.eos_token_id)[0]['generated_text']
            pred = output.split("Answer")[-1].strip().lower()
            # Normalize to option letter
            for opt in ['a', 'b', 'c', 'd']:
                if opt in pred:
                    predictions.append(opt)
                    break
            else:
                predictions.append("unknown")
            references.append(item["correct_option"].lower())
        return predictions, references

    def evaluate_general(self, predictions, references):
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        #tokenized_preds = [word_tokenize(p.lower()) for p in predictions]
        #tokenized_refs = [[[word.lower() for word in word_tokenize(r)]] for r in references]

        bleu_score = bleu.compute(predictions=predictions, references=references)
        rouge_score = rouge.compute(predictions=predictions, references=references)
        
        acc = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)) / len(predictions)

        return {
            "BLEU": bleu_score,
            "ROUGE": rouge_score,
            #"Perplexity": ppl_score,
            "Exact Match Accuracy": acc
        }

    def evaluate_mcq(self, predictions, references):
        correct = sum(p == r for p, r in zip(predictions, references))
        total = len(references)
        accuracy = correct / total
        return {"MCQ Accuracy": accuracy}

    def main(self):
        _, _, base_pipeline = self.load_model_and_tokenizer(BASE_MODEL_PATH)

        general_qa = self.load_general_qa(QA_DATASET_PATH)
        general_references = [item['answer'] for item in general_qa]

        # Base model answers
        base_preds = self.generate_general_answers(base_pipeline, general_qa)
        base_scores = self.evaluate_general(base_preds, general_references)
        print("\n== Base Model Scores ==")
        for k, v in base_scores.items():
            print(f"{k}: {v}")

        print("\nEvaluating MCQ QA...")
        mcq_qa = self.load_mcq_qa(QA_DATASET_PATH)
        mcq_references = [item['correct_option'].lower() for item in mcq_qa]

        # Base model MCQ
        base_mcq_preds, _ = self.generate_mcq_answers(base_pipeline, mcq_qa)
        base_mcq_scores = self.evaluate_mcq(base_mcq_preds, mcq_references)
        print("\n== Base Model MCQ Accuracy ==")
        print(base_mcq_scores)

        # === Save results ===
        output_data = {
            "Base_General": base_scores,
            "Base_MCQ": base_mcq_scores
        }

        os.makedirs("evaluation_results", exist_ok=True)

        # Save as JSON
        with open("evaluation_results/llm_eval_results.json", "w") as f:
            json.dump(output_data, f, indent=4)

        # Optionally, save a flat version to CSV
        with open("evaluation_results/llm_eval_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Model", "Value"])
            for model_key, scores in output_data.items():
                for metric, val in scores.items():
                    if isinstance(val, dict):  # e.g., BLEU may have multiple submetrics
                        for sub_metric, sub_val in val.items():
                            writer.writerow([sub_metric, model_key, sub_val])
                    else:
                        writer.writerow([metric, model_key, val])

        print("Evaluation results saved to JSON and CSV.")


