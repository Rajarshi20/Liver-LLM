import os
import fitz  # PyMuPDF
from transformers import AutoTokenizer
from datasets import Dataset
import json

PDF_DIR = "path/to/your/pdf_folder"
OUTPUT_DIR = "path/to/your/output_folder"
BATCH_SIZE = 100  # adjust based on memory
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text, max_tokens=512):
    sentences = text.split('. ')
    chunks, current_chunk = [], ""

    for sent in sentences:
        if len(tokenizer(current_chunk + sent).input_ids) < max_tokens:
            current_chunk += sent + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    return [{"text": chunk} for chunk in chunks if len(chunk.split()) > 10]

def batch_process_pdfs(pdf_paths, output_file):
    all_samples = []
    for pdf_path in pdf_paths:
        try:
            all_samples.extend(process_pdf(pdf_path))
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    with open(output_file, "w") as f:
        for item in all_samples:
            f.write(json.dumps(item) + "\n")

# Tokenize all chunks
def tokenize_chunks(chunks):
    tokenized = []
    for chunk in chunks:
        tokens = tokenizer(chunk, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
        tokenized.append(tokens)
    return tokenized

# Process PDFs in batches
""" all_pdfs = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
for i in range(0, len(all_pdfs), BATCH_SIZE):
    batch = all_pdfs[i:i+BATCH_SIZE]
    batch_file = os.path.join(OUTPUT_DIR, f"batch_{i//BATCH_SIZE}.jsonl")
    batch_process_pdfs(batch, batch_file)
    print(f"Saved {batch_file}") """

pdf_path = ['papers/A6-MON~1.pdf']
text = batch_process_pdfs(pdf_path, "output.json")
tokenized_chunks = tokenize_chunks(text)

# Print the first tokenized chunk
print(tokenized_chunks[0])
print(text)
