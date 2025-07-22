import re
import unicodedata
import spacy
import os
import json

nlp = spacy.load("en_core_web_sm")

class TextCleaner:
    def normalize_unicode(self, text):
        return unicodedata.normalize("NFKC", text)

    def fix_spacing_and_figs(self, text):
        text = re.sub(r'\.(?=\S)', '. ', text)
        text = re.sub(r'\s([?.!,"\'])', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)
        text = re.sub(r'\b(?:Supplementary Table|Supplementary Figure)[^\.]*\.', '', text, flags=re.IGNORECASE)
        return text.strip()

    def remove_authors_affiliations(self, text):
        patterns = [
            r'\b[A-Z][a-z]+, [A-Z]\. [A-Z]\.',
            r'\b(university|hospital|department|center|clinic|institute|school|laboratory)[^.,;]*[.,;]',
            r'www\.[^\s]+',
            r'http[s]?://[^\s]+',
            r'\b[jJ] ?med genet\b.*?\.',
            r'protected by copyright.*?technologies\.',
        ]
        for pat in patterns:
            text = re.sub(pat, '', text, flags=re.IGNORECASE | re.DOTALL)
        return text

    def remove_citations(self, text):
        text = re.sub(r'\([A-Za-z]+ et al,? \d{4}\)', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\b(doi|pmid|pmcid|fig|table)\.? ?\d*\b', '', text, flags=re.IGNORECASE)
        return text

    def remove_gene_chromosome_noise(self, text):
        text = re.sub(r'\b[A-Z0-9]{3,}\b', '', text)
        text = re.sub(r'\b\d{1,2}q\d{1,2}(\.\d+)?\b', '', text)
        text = re.sub(r'\b[cp]\d{1,2}\b', '', text)
        return text

    def remove_misc_noise(self, text):
        text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w+', '', text)
        text = re.sub(r'\(?Tel[:]?[^;]*[;.]?', '', text)
        text = re.sub(r'\(?Fax[:]?[^;]*[;.]?', '', text)
        text = re.sub(r'\d+[%\w/\-]+', '', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'[\[\]\(\)\{\}]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def clean_text(self, text):
        text = self.normalize_unicode(text)
        text = self.remove_authors_affiliations(text)
        text = self.remove_citations(text)
        text = self.remove_gene_chromosome_noise(text)
        text = self.remove_misc_noise(text)
        text = self.fix_spacing_and_figs(text)
        return text

    def symbol_density(self, text):
        symbols = re.findall(r'[^\w\s]', text)
        return len(symbols) / max(len(text), 1)

    def is_garbage_chunk(self, text):
        gene_symbols = re.findall(r'\b[A-Z0-9]{3,}\b', text)
        locus_mentions = re.findall(r'\b\d{1,2}q\d{1,2}(\.\d+)?\b', text)
        chrom_bands = re.findall(r'\b[cp]\d{1,2}\b', text)
        percent_mentions = re.findall(r'\b\d{1,3}%\b', text)

        gene_density = len(gene_symbols) / max(len(text.split()), 1)
        if gene_density > 0.3:
            return "high gene symbol density"
        if len(locus_mentions) + len(chrom_bands) > 10:
            return "excessive locus/chromosome mentions"
        if self.symbol_density(text) > 0.2 and len(percent_mentions) > 5:
            return "symbol-heavy with too many percentages"
        return None

    def is_valid_sentence(self, text, min_words=5):
        doc = nlp(text)
        has_verb = any(tok.pos_ == "VERB" for tok in doc)
        has_noun = any(tok.pos_ in ("NOUN", "PROPN") for tok in doc)
        return len(doc) >= min_words and has_verb and has_noun

class TaskCleaner:
    def __init__(self, folder="tasks", output_folder="adv_cleaned_tasks"):
        self.cleaner = TextCleaner()
        self.folder = folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def clean_text_chunks(self, chunks, log_prefix=""):
        cleaned_chunks = []
        for chunk in chunks:
            text = chunk["text"].strip()
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\.(?=\S)', '. ', text)
            text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)
            text = re.sub(r'\b(?:Supplementary Table|Supplementary Figure)[^\.]*\.', '', text, flags=re.IGNORECASE)

            text = self.cleaner.clean_text(text)

            reason = self.cleaner.is_garbage_chunk(text)
            if reason:
                print(f"[FILTERED - {log_prefix}] Reason: {reason} | Text: {text[:100]}...")
                continue

            if not self.cleaner.is_valid_sentence(text):
                print(f"[FILTERED - {log_prefix}] Reason: not a valid sentence | Text: {text[:100]}...")
                continue

            if self.cleaner.symbol_density(text) > 0.25:
                print(f"[FILTERED - {log_prefix}] Reason: high symbol density | Text: {text[:100]}...")
                continue

            cleaned_chunks.append({"text": text})
        return cleaned_chunks

    def clean_task(self, task, filename=""):
        original_count = len(task.get("chunks", []))

        for chunk in task.get("chunks", []):
            chunk["text"] = self.cleaner.clean_text(chunk["text"])

        task["chunks"] = self.clean_text_chunks(task["chunks"], log_prefix=os.path.basename(filename))
        cleaned_count = len(task["chunks"])

        print(f"[SUMMARY - {os.path.basename(filename)}] {original_count} -> {cleaned_count} chunks kept\n")
        return task

    def clean_task_file(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                cleaned_data = [self.clean_task(task, filename=filepath) for task in data]
            else:
                cleaned_data = self.clean_task(data, filename=filepath)

            out_path = os.path.join(self.output_folder, os.path.basename(filepath))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

            print(f"[SAVED] Cleaned file written to {out_path}\n")
        except Exception as e:
            print(f"[ERROR] Failed to process {filepath} - {e}")

    def run(self):
        for filename in os.listdir(self.folder):
            if filename.endswith(".json"):
                print(f"[PROCESSING] {filename}")
                self.clean_task_file(os.path.join(self.folder, filename))


# if __name__ == "__main__":
#     task_cleaner = TaskCleaner(folder="tasks")
#     task_cleaner.run()