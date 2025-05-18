import os
import fitz  # PyMuPDF
from transformers import AutoTokenizer
from datasets import Dataset
import json
import re


PDF_DIR = "papers/"
OUTPUT_DIR = "extracted_chunked_text/"
BATCH_SIZE = 1  # adjust based on memory
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

def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'-\s+', '', text)  # Fix hyphenation
    text = ' '.join(text.split())     # Remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9.,;:\-()%\s]', '', text)  # Remove unwanted chars
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.split(r'(references|acknowledgements|conflicts of interest)', text, flags=re.IGNORECASE)[0]
    
    return text

def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    text = clean_text(text)
    chunks = chunk_text(text)
    file_name = pdf_path.split('/')[1].rsplit('.', 1)[0]
    json_obj = {
        'paper_name': file_name,
        'chunks':[]
    }
    for chunk in chunks:
        if len(chunk.split()) > 10:
            json_obj['chunks'].append({"text": chunk})

    return json_obj

def batch_process_pdfs(pdf_paths, output_file):
    all_samples = []
    for pdf_path in pdf_paths:
        try:
            all_samples.append(process_pdf(pdf_path))
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    with open(output_file, "w") as f:
        for item in all_samples:
            f.write(json.dumps(item) + "\n")

# Process PDFs in batches
all_pdfs = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR)]
for i in range(0, len(all_pdfs), BATCH_SIZE):
    batch = all_pdfs[i:i+BATCH_SIZE]
    batch_file = os.path.join(OUTPUT_DIR, f"batch_{i//BATCH_SIZE}.json")
    batch_process_pdfs(batch, batch_file)
    print(f"Saved {batch_file}")

""" pdf_path = ['papers/A6-MON~1.pdf']
text = batch_process_pdfs(pdf_path, "output.json")

print(text) """
