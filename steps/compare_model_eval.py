import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from peft import PeftModel
import evaluate
import nltk
import json

from config import BASE_MODEL_PATH, LORA_MODEL_PATH, QA_DATASET_PATH

nltk.data.path.append("/hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver LLM/liverenv/lib/nltk_data/tokenizers/")
from nltk.tokenize import word_tokenize

class ModelEvaluation_compare:
    def __init__(self):
        self.MAX_NEW_TOKENS = 256
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompt_prefix = (
            "You are a helpful and knowledgeable assistant specialized in liver medicine. "
            "Based on your training and domain expertise, provide a clear and accurate answer.\n\n"
        )

    def prompt_template(self, question):
        return f"{self.prompt_prefix}Question: {question}\nAnswer:"

    def load_model_and_tokenizer(self, base_path, lora_path=None):
        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_path)
        
        if lora_path:
            print("This is the LORA Path :", lora_path)
            model = PeftModel.from_pretrained(model, lora_path, local_files_only=True)

        model.eval().to(self.DEVICE)
        pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=0 if self.DEVICE == "cuda" else -1
        )
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
            prompt = self.prompt_template(item['question'])
            output = pipeline(prompt, max_new_tokens=self.MAX_NEW_TOKENS, pad_token_id=pipeline.tokenizer.eos_token_id)[0]['generated_text']
            answer = output.split("Answer:")[-1].strip()
            predictions.append(answer)
        return predictions

    def generate_mcq_answers(self, pipeline, dataset):
        predictions, references = [], []
        for item in dataset:
            prompt = self.prompt_prefix + f"Question: {item['question']}\n"
            for key, value in item['options'].items():
                prompt += f"{key}. {value}\n"
            prompt += "Answer (a/b/c/d):"
            output = pipeline(prompt, max_new_tokens=10, pad_token_id=pipeline.tokenizer.eos_token_id)[0]['generated_text']
            pred = output.split("Answer")[-1].strip().lower()
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
        bleu_score = bleu.compute(predictions=predictions, references=references)
        rouge_score = rouge.compute(predictions=predictions, references=references)
        acc = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)) / len(predictions)
        return {
            "BLEU": bleu_score,
            "ROUGE": rouge_score,
            "Exact Match Accuracy": acc
        }

    def evaluate_mcq(self, predictions, references):
        correct = sum(p == r for p, r in zip(predictions, references))
        total = len(references)
        accuracy = correct / total
        return {"MCQ Accuracy": accuracy}

    def main(self):
        print("Evaluating Fine-Tuned Model...\n")
        _, _, finetuned_pipeline = self.load_model_and_tokenizer(BASE_MODEL_PATH, LORA_MODEL_PATH)
        _, _, base_pipeline = self.load_model_and_tokenizer(BASE_MODEL_PATH)

        general_qa = self.load_general_qa(QA_DATASET_PATH)
        general_references = [item['answer'] for item in general_qa]

        # Fine-tuned model answers
        finetuned_preds = self.generate_general_answers(finetuned_pipeline, general_qa)
        finetuned_scores = self.evaluate_general(finetuned_preds, general_references)
        print("== Fine-Tuned Model Scores ==")
        for k, v in finetuned_scores.items():
            print(f"{k}: {v}")

        # Base model answers
        base_preds = self.generate_general_answers(base_pipeline, general_qa)
        base_scores = self.evaluate_general(base_preds, general_references)
        print("\n== Base Model Scores ==")
        for k, v in base_scores.items():
            print(f"{k}: {v}")

        print("\nEvaluating MCQ QA...")
        mcq_qa = self.load_mcq_qa(QA_DATASET_PATH)
        mcq_references = [item['correct_option'].lower() for item in mcq_qa]

        # Fine-tuned model MCQ
        ft_mcq_preds, _ = self.generate_mcq_answers(finetuned_pipeline, mcq_qa)
        ft_mcq_scores = self.evaluate_mcq(ft_mcq_preds, mcq_references)
        print("\n== Fine-Tuned Model MCQ Accuracy ==")
        print(ft_mcq_scores)

        # Base model MCQ
        base_mcq_preds, _ = self.generate_mcq_answers(base_pipeline, mcq_qa)
        base_mcq_scores = self.evaluate_mcq(base_mcq_preds, mcq_references)
        print("\n== Base Model MCQ Accuracy ==")
        print(base_mcq_scores)

