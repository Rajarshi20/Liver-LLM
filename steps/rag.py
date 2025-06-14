from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
import os

class RetrievalAugmentedGeneration:
    def __init__(self, uploaded_file_content, model_path="meta-llama/Meta-Llama-3-8B-Instruct"):
        # Getting file and query uploaded
        self.uploaded_file = uploaded_file_content

        # Use e5-base embedding model (instruction-tuned)
        self.embedding_model = SentenceTransformer("intfloat/e5-base")

        # Load LLM model and tokenizer
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.llm_model.eval()

        self.device = 0 if torch.cuda.is_available() else -1
        self.llm_pipeline = pipeline("text-generation", model=self.llm_model, tokenizer=self.llm_tokenizer, device=self.device)

        # Setup Chroma client
        self.chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=".chroma_db"))
        self.collection = self.chroma_client.get_or_create_collection(name="liver_papers")

    def extract_text_from_pdf(self, chunk_size=500):
        try:
            doc = fitz.open(stream=self.uploaded_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            return chunks
            # doc = fitz.open(pdf_path)
            # text = ""
            # for page in doc:
            #     text += page.get_text()
            # doc.close()
        except Exception as e:
            raise ValueError(f"Error reading PDF: {e}")

    def build_chroma_index(self, chunks):
        #self.collection.delete(where={})  # Clear old chunks if re-indexing
        embeddings = self.embedding_model.encode([f"passage: {c}" for c in chunks])
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.collection.add(documents=[chunk], embeddings=[embedding.tolist()], ids=[f"chunk_{i}"])

    def retrieve_chunks(self, query, k=5):
        query_embedding = self.embedding_model.encode(f"query: {query}").tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        return results['documents'][0] if results['documents'] else []

    def generate_response(self, query, retrieved_chunks):
        context = "\n".join(retrieved_chunks)[:4000]  # ensure token budget
        prompt = f"Use the following liver medical research context to answer or summarize:\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        response = self.llm_pipeline(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]['generated_text']
        return response.replace(prompt, "").strip()
    
    def main(self, query):
        chunks = self.extract_text_from_pdf()
        self.build_chroma_index(chunks)
        retrieved = self.retrieve_chunks(query)
        response = self.generate_response(query, retrieved_chunks=retrieved)
        # print(response)
        return response

# Example usage:
# rag = RetrievalAugmentedGeneration()
# chunks = rag.extract_text_from_pdf("./../papers/Hepatocellular_carcinoma.pdf")
# rag.build_chroma_index(chunks)
# query = "Is blood transfusion associated with recurrence of hepatocellular carcinoma after hepatectomy in Child-Pugh class A patients?"
# retrieved = rag.retrieve_chunks(query)
# response = rag.generate_response(query, retrieved)
# print(response)