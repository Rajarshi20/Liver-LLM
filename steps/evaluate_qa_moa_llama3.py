# Evaluating meta-llama/Meta-Llama-3-70B-Instruct

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
import os
os.environ["HF_MODULES_CACHE"] = "/path_to_GPU/scratch/huggingface_modules"
os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)
os.environ["HF_EVALUATE_CACHE"] = "/path_to_GPU/scratch/evaluate_cache"
os.makedirs("/path_to_GPU/scratch/evaluate_cache", exist_ok=True)
os.environ["HF_HOME"] = "/path_to_GPU/home/Liver-LLM/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/path_to_GPU/home/Liver-LLM/hf_cache"
import evaluate
import nltk
nltk.download("punkt", download_dir="/path_to_GPU/scratch/nltk_data")

from nltk.tokenize import word_tokenize
import json
from config import BASE_MODEL_PATH, QA_BASEMODEL_MOA_EVAL

class Llama3_Base_Evaluation:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model_and_tokenizer(self, base_path):
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
                base_path,
                quantization_config=bnb_config,
                device_map={"": device_id},
                trust_remote_code=True,
            )
        #base_model = AutoModelForCausalLM.from_pretrained(base_path)
        # ft_model = PeftModel.from_pretrained(model, lora_path, local_files_only=True)
        # ft_model = ft_model.merge_and_unload()
        # ft_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        # Define generation pipeline with controlled decoding
        pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            num_beams=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
        return tokenizer, model, pipeline

    def load_moa_qa(self, path):
        moa_qa = []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for obj in data:
                # line = line.strip()
                # if not line:
                #     continue
                # obj = json.loads(line)
                # if obj.get("type") == "MOA":
                moa_qa.append({
                    "question": obj["question"], 
                    "answer": obj["reference_answer"], 
                    "predicted_answer": obj["predicted_answer"]
                })
        return moa_qa

    def generate_moa_answers(self, pipeline, dataset):
        predictions = []
        for item in dataset:
            prompt = (
                f"### INSTRUCTION:\n"
                f"You are a helpful medical assistant trained on liver diseases.\n"
                f"Study and respond accurately and concisely to the question below.\n"
                f"### INPUT:\n"
                f"Question: {item['question']}\n"
                f"### OUTPUT:\n"
                f"Answer:"
            )
            output = pipeline(prompt)[0]['generated_text']
            answer = output.split("Answer:")[-1].strip()

            # Optional: stop at first hallucinated section if "###" or extra prompt shows up
            for stop_token in ["###", "\n\n", "\nInstruction", "Instruction:", "Input:"]:
                if stop_token in answer:
                    answer = answer.split(stop_token)[0].strip()

            # Optional: truncate hallucinated trailing text after first period.
            """ if "." in answer:
                answer = answer.split(".")[0].strip() + "." """

            predictions.append(answer)
        return predictions

    def evaluate_moa(self, predictions, references):
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        bertscore = evaluate.load("bertscore")

        bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")
        bleu_score = bleu.compute(predictions=predictions, references=references)
        rouge_score = rouge.compute(predictions=predictions, references=references)
        exact_match = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)) / len(predictions)
        avg_bertscore_f1 = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])

        return {
            "BLEU": bleu_score,
            "ROUGE": rouge_score,
            "BERT SCORE":avg_bertscore_f1,
            "Exact Match Accuracy": exact_match
        }

    def main(self):

        output_dir = "/path_to_GPU/home/eval_outputs/llama3_base_eval"
        os.makedirs(output_dir, exist_ok=True)
        print("Loading model and tokenizer...")
        tokenizer, model, pipeline = self.load_model_and_tokenizer(BASE_MODEL_PATH)

        print("Loading MOA validation dataset...")
        moa_qa = self.load_moa_qa(QA_BASEMODEL_MOA_EVAL)
        print(f"Loaded {len(moa_qa)} MOA questions")

        print("Generating predictions...")
        predictions = self.generate_moa_answers(pipeline, moa_qa)
        references = [item["answer"] for item in moa_qa]
        ft_predicted_answers = [item["predicted_answer"] for item in moa_qa]

        print("Saving output...")
        results = [
            {
                "question": item["question"],
                "predicted_answer": ft_pred,
                "base_llama_answer": pred,
                "reference_answer": ref
            }
            for item, ft_pred, pred, ref in zip(moa_qa, ft_predicted_answers, predictions, references)
        ]
        with open(os.path.join(output_dir, "llama3_base_moa_predictions.json"), "w") as f:
            json.dump(results, f, indent=2)

        print("Evaluating predictions...")
        scores = self.evaluate_moa(predictions, ft_predicted_answers)

        with open(os.path.join(output_dir, "llama3_base_moa_qa_scores.json"), "w") as f:
            json.dump(scores, f, indent=2)

        print("\nEvaluation Scores:")
        for k, v in scores.items():
            print(f"{k}: {v}")