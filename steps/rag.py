from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import fitz  # PyMuPDF
import faiss
import numpy as np
import os

class RetrievalAugmentedGeneration:
    def __init__(self, model_path="./liver-llm"):
        # Load embedding model
        self.embedding_model = SentenceTransformer("deepseek-ai/deepseek-embedding")

        # Load LLama-based Liver LLM locally
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.llm_model.eval()

        # Set device and initialize pipeline
        self.device = 0 if torch.cuda.is_available() else -1
        self.llm_pipeline = pipeline("text-generation", model=self.llm_model, tokenizer=self.llm_tokenizer, device=self.device)

    # Extract and chunk text from PDF
    def extract_text_from_pdf(self, pdf_path, chunk_size=500):
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            raise ValueError(f"Error reading PDF: {e}")

        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        return chunks 

    # Embed and index chunks
    def build_faiss_index(self, chunks):
        embeddings = self.embedding_model.encode(chunks)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index, embeddings

    # Retrieve top-k relevant chunks
    def retrieve_chunks(self, query, chunks, embeddings, index, k=5):
        query_vec = self.embedding_model.encode([query])
        _, indices = index.search(query_vec, k)
        return [chunks[i] for i in indices[0]]

    # Generate answer or summary from retrieved context
    def generate_response(self, query, retrieved_chunks):
        context = "\n".join(retrieved_chunks)[:4000]  # truncate to fit token limit
        prompt = f"Use the following liver medical research context to answer or summarize:\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        response = self.llm_pipeline(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]['generated_text']
        return response.replace(prompt, "").strip()

# Example usage:
# rag = RetrievalAugmentedGeneration(model_path="./liver-llm")
# chunks = rag.extract_text_from_pdf("example_liver_paper.pdf")
# index, embeddings = rag.build_faiss_index(chunks)
# query = "Summarize the findings related to liver fibrosis."
# retrieved = rag.retrieve_chunks(query, chunks, embeddings, index)
# response = rag.generate_response(query, retrieved)
# print(response)
