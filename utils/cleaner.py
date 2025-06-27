import re
import json
import os
import unicodedata
import spacy

nlp = spacy.load("en_core_web_sm")

def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text)

def fix_spacing(text):
    text = re.sub(r'\.(?=\S)', '. ', text)
    text = re.sub(r'\s([?.!,"\'])', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_authors_affiliations(text):
    patterns = [
        r'\b[A-Z][a-z]+, [A-Z]\. [A-Z]\.',   # Author names
        r'\b(university|hospital|department|center|clinic|institute|school|laboratory)[^.,;]*[.,;]', # affiliations
        r'www\.[^\s]+',                      # URLs
        r'http[s]?://[^\s]+',                # URLs
        r'\b[jJ] ?med genet\b.*?\.',         # journal mentions
        r'protected by copyright.*?technologies\.',  # copyright
    ]
    for pat in patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE | re.DOTALL)
    return text

def remove_citations(text):
    text = re.sub(r'\([A-Za-z]+ et al,? \d{4}\)', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\b(doi|pmid|pmcid|fig|table)\.? ?\d*\b', '', text, flags=re.IGNORECASE)
    return text

def remove_gene_chromosome_noise(text):
    text = re.sub(r'\b[A-Z0-9]{3,}\b', '', text)
    text = re.sub(r'\b\d{1,2}q\d{1,2}(\.\d+)?\b', '', text)
    text = re.sub(r'\b[cp]\d{1,2}\b', '', text)
    return text

def remove_misc_noise(text):
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w+', '', text)
    text = re.sub(r'\(?Tel[:]?[^;]*[;.]?', '', text)
    text = re.sub(r'\(?Fax[:]?[^;]*[;.]?', '', text)
    text = re.sub(r'\d+[%\w/\-]+', '', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'[\[\]\(\)\{\}]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_text(text):
    text = normalize_unicode(text)
    text = remove_authors_affiliations(text)
    text = remove_citations(text)
    text = remove_gene_chromosome_noise(text)
    text = remove_misc_noise(text)
    text = fix_spacing(text)
    return text

# Your original symbol density and garbage chunk detection functions unchanged
def symbol_density(text):
    symbols = re.findall(r'[^\w\s]', text)
    return len(symbols) / max(len(text), 1)

def is_garbage_chunk(text):
    gene_symbols = re.findall(r'\b[A-Z0-9]{3,}\b', text)
    locus_mentions = re.findall(r'\b\d{1,2}q\d{1,2}(\.\d+)?\b', text)
    chrom_bands = re.findall(r'\b[cp]\d{1,2}\b', text)
    percent_mentions = re.findall(r'\b\d{1,3}%\b', text)

    gene_density = len(gene_symbols) / max(len(text.split()), 1)
    if gene_density > 0.3:
        return "high gene symbol density"
    if len(locus_mentions) + len(chrom_bands) > 10:
        return "excessive locus/chromosome mentions"
    if symbol_density(text) > 0.2 and len(percent_mentions) > 5:
        return "symbol-heavy with too many percentages"
    return None

def is_valid_sentence(text, min_words=5):
    doc = nlp(text)
    has_verb = any(tok.pos_ == "VERB" for tok in doc)
    has_noun = any(tok.pos_ in ("NOUN", "PROPN") for tok in doc)
    return len(doc) >= min_words and has_verb and has_noun

def clean_text_chunks(chunks, log_prefix=""):
    cleaned_chunks = []
    for chunk in chunks:
        text = chunk["text"].strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.(?=\S)', '. ', text)
        text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)
        text = re.sub(r'\b(?:Supplementary Table|Supplementary Figure)[^\.]*\.', '', text, flags=re.IGNORECASE)

        # Apply your integrated cleaning function here
        text = clean_text(text)

        reason = is_garbage_chunk(text)
        if reason:
            print(f"[FILTERED - {log_prefix}] Reason: {reason} | Text: {text[:100]}...")
            continue

        if not is_valid_sentence(text):
            print(f"[FILTERED - {log_prefix}] Reason: not a valid sentence | Text: {text[:100]}...")
            continue

        if symbol_density(text) > 0.25:
            print(f"[FILTERED - {log_prefix}] Reason: high symbol density | Text: {text[:100]}...")
            continue

        cleaned_chunks.append({"text": text})
    return cleaned_chunks

def clean_task(task, filename=""):
    original_count = len(task.get("chunks", []))

    for chunk in task.get("chunks", []):
        first_clean = clean_text(chunk["text"])
        chunk["text"] = first_clean

    task["chunks"] = clean_text_chunks(task["chunks"], log_prefix=os.path.basename(filename))
    cleaned_count = len(task["chunks"])

    print(f"[SUMMARY - {os.path.basename(filename)}] {original_count} -> {cleaned_count} chunks kept\n")
    return task

def clean_task_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            cleaned_data = [clean_task(task, filename=filepath) for task in data]
        else:
            cleaned_data = clean_task(data, filename=filepath)

        os.makedirs("adv_cleaned_tasks", exist_ok=True)
        out_path = os.path.join("adv_cleaned_tasks", os.path.basename(filepath))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        print(f"[SAVED] Cleaned file written to {out_path}\n")
    except Exception as e:
        print(f"[ERROR] Failed to process {filepath} - {e}")

if __name__ == "__main__":
    folder = "tasks"
    os.makedirs("adv_cleaned_tasks", exist_ok=True)
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            print(f"[PROCESSING] {filename}")
            clean_task_file(os.path.join(folder, filename))
