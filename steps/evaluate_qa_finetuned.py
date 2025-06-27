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


from config import BASE_MODEL_PATH, QA_FINETUNED_MOA_MODEL_PATH, QA_FINETUNED_MOA_VALIDATION_DATASET

class FineTunedModelEvaluation:
    def __init__(self):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model_and_tokenizer(self, base_path, lora_path):
        print("Loading tokenizer and LoRA model from:", lora_path)
        base_model = AutoModelForCausalLM.from_pretrained(base_path)
        model = PeftModel.from_pretrained(base_model, lora_path, local_files_only=True)
        model.eval().to(self.DEVICE)

        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        # Define generation pipeline with controlled decoding
        pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=0 if self.DEVICE == "cuda" else -1,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            num_beams=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        return tokenizer, model, pipeline

    def load_moa_qa(self, path):
        moa_qa = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") == "MOA":
                    moa_qa.append({"question": obj["question"], "answer": obj["answer"]})
        return moa_qa

    def generate_moa_answers(self, pipeline, dataset):
        predictions = []
        for item in dataset:
            prompt = (
                f"### INSTRUCTION:\n"
                f"You are a helpful medical assistant trained on liver diseases.\n"
                f"Study and respond accurately to the question below.\n"
                f"### INPUT:\n"
                f"Question: {item['question']}\n"
                f"### OUTPUT:\n"
                f"Answer:"
            )
            output = pipeline(prompt)[0]['generated_text']
            answer = output.split("Answer:")[-1].strip()

            # Optional: truncate hallucinated trailing text after first period.
            if "." in answer:
                answer = answer.split(".")[0].strip() + "."

            predictions.append(answer)
        return predictions

    def evaluate_moa(self, predictions, references):
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")

        bleu_score = bleu.compute(predictions=predictions, references=references)
        rouge_score = rouge.compute(predictions=predictions, references=references)
        exact_match = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)) / len(predictions)

        return {
            "BLEU": bleu_score,
            "ROUGE": rouge_score,
            "Exact Match Accuracy": exact_match
        }

    def main(self):
        print("Loading model and tokenizer...")
        tokenizer, model, pipeline = self.load_model_and_tokenizer(BASE_MODEL_PATH, QA_FINETUNED_MOA_MODEL_PATH)

        print("Loading MOA validation dataset...")
        moa_qa = self.load_moa_qa(QA_FINETUNED_MOA_VALIDATION_DATASET)
        print(f"Loaded {len(moa_qa)} MOA questions")

        print("Generating predictions...")
        predictions = self.generate_moa_answers(pipeline, moa_qa)
        references = [item["answer"] for item in moa_qa]

        print("Evaluating predictions...")
        scores = self.evaluate_moa(predictions, references)

        print("Saving output...")
        results = [
            {
                "question": item["question"],
                "predicted_answer": pred,
                "reference_answer": ref
            }
            for item, pred, ref in zip(moa_qa, predictions, references)
        ]
        with open("moa_qa_predictions.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\nEvaluation Scores:")
        for k, v in scores.items():
            print(f"{k}: {v}")
