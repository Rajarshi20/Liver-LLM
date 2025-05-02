import fitz
from pdf2image import convert_from_path
import cv2
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

# Extract images from PDF
def extract_images_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    return images  # List of PIL.Image objects

# Process images using OpenCV (example)
def process_images(images):
    for img in images:
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # Process each image as needed

# Example usage
pdf_path = 'path_to_your_pdf.pdf'
text = extract_text_from_pdf(pdf_path)
images = extract_images_from_pdf(pdf_path)
process_images(images)
