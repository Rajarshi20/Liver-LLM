import fitz
# from pdf2image import convert_from_path
# import cv2
import numpy as np

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    doc.close()
    return text

pdf_path = 'A6-MON~1.pdf'
text = extract_text_from_pdf(pdf_path)

print(text)
