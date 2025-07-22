from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig
import torch
import fitz  # PyMuPDF
import faiss
import numpy as np
import os
from config import BASE_MODEL_PATH, QA_FINETUNED_MOA_MODEL_PATH
from utils import TextCleaner

class RetrievalAugmentedGeneration:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Device setup
        self.device = 0 if torch.cuda.is_available() else -1
        device_str = "cuda" if self.device == 0 else "cpu"

        # Load tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device_str)

        lora_path = QA_FINETUNED_MOA_MODEL_PATH
        if lora_path:
            self.llm_model = PeftModel.from_pretrained(base_model, lora_path).to(device_str)
        else:
            self.llm_model = base_model

        self.llm_model.eval()
        self.llm_pipeline = pipeline("text-generation", model=self.llm_model, tokenizer=self.tokenizer, device=self.device)

    def extract_text_from_pdf(self, pdf_file, chunk_size=500):
        if not os.path.isfile(pdf_file):
            raise FileNotFoundError(f"PDF file not found: {pdf_file}")
        else:
            doc = fitz.open(pdf_file)
            text = "".join([page.get_text() for page in doc])
            
            cleaner = TextCleaner()
            text = cleaner.clean_text(text)
            
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            return chunks

    def build_faiss_index(self, chunks):
        embeddings = self.embedding_model.encode(chunks)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index, embeddings

    def retrieve_chunks(self, query, chunks, embeddings, index, k=5):
        query_vec = self.embedding_model.encode([query])
        _, indices = index.search(query_vec, k)
        return [chunks[i] for i in indices[0]]

    def generate_response(self, query, retrieved_chunks):
        # 1. Join chunks and limit total context length
        context = "\n".join(retrieved_chunks)[:1000]

        # 2. Construct prompt
        prompt = (
            f"### INSTRUCTION:\n"
            f"You are a helpful medical assistant trained on liver diseases.\n"
            f"Use the following liver medical research CONTEXT to answer the QUESTION.\n\n"
            f"### CONTEXT:\n"
            f"{context}\n\n"
            f"### QUESTION:\n{query}\n\n"
            f"Answer:"
        )

        # 3. Generate with stable parameters (no sampling, shorter output, eos_token_id)
        response = self.llm_pipeline(
            prompt,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.5,
            eos_token_id=self.tokenizer.eos_token_id,
        )[0]['generated_text']

        print('[LLM RESPONSE]: ', response)
        print()

        # 4. Clean output
        answer_raw = response.replace(prompt, "").strip()
        # print('RAW RESP: ', answer_raw)

        # 5. Stop at double newlines or use deduplication
        answer_trimmed = answer_raw.split("\n\n")[0].strip()
        answer_clean = self.remove_repeated_sentences(answer_trimmed)

        # print('CLEANED: ', answer_clean)
        return answer_clean
    
    def save_faiss_index(self, index, path="index.faiss"):
        faiss.write_index(index, path)

    def load_faiss_index(self, path="index.faiss"):
        return faiss.read_index(path)

    def remove_repeated_sentences(self, text):
        import re
        # Split using regex for robustness
        sentences = re.split(r'(?<=[.!?]) +', text)
        seen = set()
        result = []

        for s in sentences:
            s_clean = s.strip()
            if s_clean and s_clean not in seen:
                result.append(s_clean)
                seen.add(s_clean)

        return " ".join(result)