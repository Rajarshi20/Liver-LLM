import os
import fitz  # PyMuPDF
from transformers import AutoTokenizer
import json
import re
import logging

class ExtractPaper:
    PDF_DIR = "papers/"
    OUTPUT_DIR = "extracted_chunked_text/"
    BATCH_SIZE = 1  # adjust based on memory
    MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Extract text from PDF
    def extract_text_from_pdf(self, pdf_path):
        text= ""
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logging.warning(f"Failed to open PDF: {pdf_path} | Error: {e}")
            return ""  # Skip file if cannot open

        try:
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    text += page_text
                except Exception as e:
                    logging.warning(f"Failed to extract text from page {page_num} in {pdf_path} | Error: {e}")
        finally:
            doc.close()

        if not text.strip():
            logging.info(f"No text found in PDF: {pdf_path}")
        return text

    def chunk_text(self, text, max_tokens=512):
        sentences = text.split('. ')
        chunks, current_chunk = [], ""

        for sent in sentences:
            if len(self.tokenizer(current_chunk + sent).input_ids) < max_tokens:
                current_chunk += sent + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sent + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def clean_text(self, text):
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'-\s+', '', text)  # Fix hyphenation
        text = ' '.join(text.split())     # Remove extra spaces
        text = re.sub(r'[^a-zA-Z0-9.,;:\-()%\s]', '', text)  # Remove unwanted chars
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.split(r'(references|acknowledgements|conflicts of interest)', text, flags=re.IGNORECASE)[0]
        
        return text

    def process_pdf(self, pdf_path):
        text = self.extract_text_from_pdf(pdf_path=pdf_path)
        text = self.clean_text(text)
        chunks = self.chunk_text(text)
        # file_name = path.split('/')[1].rsplit('.', 1)[0]
        file_name = os.path.splitext(os.path.basename(pdf_path))[0]
        json_obj = {
            'paper_name': file_name,
            'chunks':[]
        }
        for chunk in chunks:
            if len(chunk.split()) > 10:
                json_obj['chunks'].append({"text": chunk})

        return json_obj

    def batch_process_pdfs(self, pdf_paths, output_file):
        all_samples = []
        for pdf_path in pdf_paths:
            try:
                all_samples.append(self.process_pdf(pdf_path=pdf_path))
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
        # print(all_samples)
        # print()
        with open(output_file, "w") as f:
            for item in all_samples:
                f.write(json.dumps(item) + "\n")
        # with open(output_file, "w") as f:
        #     json.dump(all_samples, f, indent=2)

    def main(self):
        # Processing PDFs in batches

        all_pdfs = [os.path.join(self.PDF_DIR, f) for f in os.listdir(self.PDF_DIR)]
        for i in range(0, len(all_pdfs), self.BATCH_SIZE):
            batch = all_pdfs[i : i+self.BATCH_SIZE]

            os.makedirs(self.OUTPUT_DIR, exist_ok=True)
            batch_file = os.path.join(self.OUTPUT_DIR, f"batch_{i//self.BATCH_SIZE}.json")
            
            self.batch_process_pdfs(pdf_paths=batch, output_file=batch_file)
            print(f"Saved {batch_file}")

        """ pdf_path = ['papers/A6-MON~1.pdf']
        text = batch_process_pdfs(pdf_path, "output.json")

        print(text) """
