import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import BASE_MODEL_PATH, LORA_MODEL_PATH
from peft import PeftModel


DATA_DIR = "extracted_chunks_text/"  # ← change this
OUTPUT_QA_PATH = "generated_liver_qa.jsonl"

# 1. Load model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH,local_files_only=True).cuda()
model.enable_input_require_grads()
model.eval()

# 2. QA generation prompt
PROMPT_TEMPLATE = """You are a clinical assistant trained on liver cancer literature. 
Given the following text, generate one high-quality question-answer pair relevant to liver cancer diagnosis, treatment, or research.

Text:
{text}

Output:
Q: <question>
A: <answer>
"""

def generate_qa(text_chunk, max_new_tokens=200):
    prompt = PROMPT_TEMPLATE.format(text=text_chunk)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Q:" in output_text and "A:" in output_text:
        try:
            q = output_text.split("Q:")[1].split("A:")[0].strip()
            a = output_text.split("A:")[1].strip()
            return {"question": q, "answer": a}
        except:
            return None
    return None

def process_all_batches(data_dir, output_path):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])

    with open(output_path, "w") as out_file:
        for file in tqdm(files, desc="Processing batch files"):
            file_path = os.path.join(data_dir, file)
            with open(file_path, "r") as f:
                papers = json.load(f)

            for paper in papers:
                for chunk in paper.get("chunks", []):
                    if 100 < len(chunk) < 1500:
                        qa = generate_qa(chunk)
                        if qa:
                            json.dump(qa, out_file)
                            out_file.write("\n")

if __name__ == "__main__":
    process_all_batches(DATA_DIR, OUTPUT_QA_PATH)
