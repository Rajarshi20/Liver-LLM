import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from peft import PeftModel
from datasets import load_dataset
import evaluate
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


# -------------------------------
# CONFIGURATION
# -------------------------------
BASE_MODEL_PATH = "meta-llama/Llama-2-7b-hf"
LORA_MODEL_PATH = "./lora-liver-finetuned"
QA_DATASET_PATH = "./qa_dataset.json"  # or load from HuggingFace
MAX_NEW_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# MODEL & TOKENIZER LOADING
# -------------------------------
def load_model_and_tokenizer(base_path, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure valid pad token

    base_model = AutoModelForCausalLM.from_pretrained(base_path)
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval().to(DEVICE)

    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1)
    return tokenizer, model, pipeline


# -------------------------------
# DATA LOADING
# -------------------------------
def load_qa_dataset(path):
    import json
    with open(path, "r") as f:
        data = json.load(f)
    return [{"question": item["question"], "answer": item["answer"]} for item in data]


# -------------------------------
# TEXT GENERATION
# -------------------------------
def generate_answers(pipeline, dataset, max_tokens):
    predictions = []
    for item in dataset:
        prompt = f"Question: {item['question']}\nAnswer:"
        output = pipeline(prompt, max_new_tokens=max_tokens, pad_token_id=pipeline.tokenizer.eos_token_id)[0]['generated_text']
        answer = output.split("Answer:")[-1].strip()
        predictions.append(answer)
    return predictions


# -------------------------------
# EVALUATION METRICS
# -------------------------------
def evaluate_outputs(predictions, references):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    perplexity = evaluate.load("perplexity", module_type="metric")

    # Tokenize
    tokenized_preds = [word_tokenize(p.lower()) for p in predictions]
    tokenized_refs = [[word_tokenize(r.lower())] for r in references]

    bleu_score = bleu.compute(predictions=tokenized_preds, references=tokenized_refs)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    
    # Flatten texts for perplexity (optional - only works on short text, token limit ~512)
    ppl_dataset = [{"text": pred} for pred in predictions]
    ppl_score = perplexity.compute(data=ppl_dataset, model_id=BASE_MODEL_PATH)

    # Accuracy (exact match)
    acc = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)) / len(predictions)

    return {
        "BLEU": bleu_score,
        "ROUGE": rouge_score,
        "Perplexity": ppl_score,
        "Accuracy": acc
    }


# -------------------------------
# MAIN SCRIPT
# -------------------------------
def main():
    print("Loading model and tokenizer...")
    tokenizer, model, pipeline = load_model_and_tokenizer(BASE_MODEL_PATH, LORA_MODEL_PATH)

    print("Loading dataset...")
    dataset = load_qa_dataset(QA_DATASET_PATH)
    questions = [item["question"] for item in dataset]
    references = [item["answer"] for item in dataset]

    print("Generating answers...")
    predictions = generate_answers(pipeline, dataset, MAX_NEW_TOKENS)

    print("Evaluating...")
    scores = evaluate_outputs(predictions, references)

    print("\nEvaluation Results:")
    for k, v in scores.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
