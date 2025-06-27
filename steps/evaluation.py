import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from peft import PeftModel
from datasets import load_dataset
import evaluate
import nltk
#nltk.download("punkt_tab")

nltk.data.path.append("/hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver LLM/liverenv/lib/nltk_data/tokenizers/")

from nltk.tokenize import word_tokenize
import json


from config import BASE_MODEL_PATH, LORA_MODEL_PATH, QA_DATASET_PATH

class ModelEvaluation:
    def __init__(self):
        self.MAX_NEW_TOKENS = 128
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model_and_tokenizer(self, base_path, lora_path):
        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(base_path)
        print("This is the LORA Path :",lora_path)
        model = PeftModel.from_pretrained(base_model, lora_path,local_files_only=True)
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
        print("Loading model and tokenizer...")
        tokenizer, model, pipeline = self.load_model_and_tokenizer(BASE_MODEL_PATH, LORA_MODEL_PATH)

        print("Evaluating General QA...")
        general_qa = self.load_general_qa(QA_DATASET_PATH)
        print("QA loaded from the dataset ", general_qa[1])
        general_predictions = self.generate_general_answers(pipeline, general_qa)
        print("Generated predictions from model: ",general_predictions[1])
        general_references = [item['answer'] for item in general_qa]
        print("Answer from the QA dataset: ",general_references[1])
        general_scores = self.evaluate_general(general_predictions, general_references)
        for k, v in general_scores.items():
            print(f"{k}: {v}")

        print("\nEvaluating MCQ QA...")
        mcq_qa = self.load_mcq_qa(QA_DATASET_PATH)
        print("MCQ data from dataset: ",mcq_qa[1])
        mcq_predictions, mcq_references = self.generate_mcq_answers(pipeline, mcq_qa)
        print("Predicted Option: ",mcq_predictions)
        print("Actual Option: ",mcq_references)
        mcq_scores = self.evaluate_mcq(mcq_predictions, mcq_references)
        for k, v in mcq_scores.items():
            print(f"{k}: {v}")

