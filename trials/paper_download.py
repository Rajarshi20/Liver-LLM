import requests
import os
import pandas as pd
import math

class DownloadPapers:
    INPUT_DIR = 'source_csv'
    CLEANED_CSV_DIR = 'cleaned_csv'
    LOGS_OUTPUT_DIR = 'paper_extraction_results/logs'
    PAPERS_OUTPUT_DIR = 'paper_extraction_results/papers'

    """
    Purpose: Creates the directories if they don't exist.
    """
    def create_directory(self):
        os.makedirs(self.INPUT_DIR, exist_ok=True)
        os.makedirs(self.CLEANED_CSV_DIR, exist_ok=True)
        os.makedirs(self.LOGS_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.PAPERS_OUTPUT_DIR, exist_ok=True)

    """
    Purpose: Reads all the hop csv files and cleans them - removes duplicate DOIs and rows with missing DOIs.
    """
    def read_and_clean_csvs(self):
        for file in os.listdir(self.INPUT_DIR):
            print(file)
            file_without_extension = file.split('.')[0]
            df = pd.read_csv(f'{self.INPUT_DIR}/{file}')
            print(df.head())

            df = df.dropna(subset=['DOI'])
            df = df.drop_duplicates(subset=['DOI'])

            # Splitting the big CDV into smaller chunks
            chunk_size = 5000
            num_chunks = math.ceil(len(df) / chunk_size)
            for i in range(num_chunks):
                chunk = df[i*chunk_size: (i+1)*chunk_size]
                chunk_csv_file = f'{file_without_extension}_doi_subset_{i+1:03d}_of_{num_chunks:03d}.csv'
                chunk_path = os.path.join(self.CLEANED_CSV_DIR, chunk_csv_file)
                chunk.to_csv(chunk_path, index=False)
                print(f'Saved {chunk_path}')

    """
    Purpose: Getting the Open Access URL of the paper using it's DOI. 
    """
    def get_open_access_pdf_url(self, doi):
        headers = {
            'Accept': 'application/json'
        }
        email = 'st191452@stud.uni-stuttgart.de'
        url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                best_oa_location = data.get('best_oa_location')
                if best_oa_location and best_oa_location.get('url_for_pdf'):
                    return best_oa_location['url_for_pdf']
        except Exception as e:
            print("Exception: ", e)
            return None

    """
    Purpose: Using the PDF URL to stream and save the paper in the specified output location
    """
    def download_pdf(self, pdf_url, output_path):
        try:
            response = requests.get(pdf_url, stream=True, timeout=30)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                print(f"PDF downloaded successfully: {output_path}")
                return 200
            else:
                print("Failed to download PDF.")
                return 500
        except Exception as e:
            print("Exception: ", e)
            return 500

    def download_paper_by_doi(self, doi, title, csv_file):
        # Downloading the papers from each chunk CSV file into a separate directory
        output_path = os.path.join(self.PAPERS_OUTPUT_DIR, csv_file)
        
        # Creating directory if needed
        os.makedirs(output_path, exist_ok=True)
        
        failed_downloads_path = f'{self.LOGS_OUTPUT_DIR}/log_{csv_file}.csv'

        if os.path.exists(failed_downloads_path):
            failed_downloads = pd.read_csv(failed_downloads_path).to_dict('records')
        else:
            failed_downloads = []
        
        print(f"Looking up DOI: {doi}")
        if doi:
            pdf_url = self.get_open_access_pdf_url(doi=doi)
            if pdf_url:
                filename = doi.replace('/', '_') + '.pdf'
                output_path = os.path.join(output_path, filename)
                res = self.download_pdf(pdf_url, output_path)
                if res == 500:
                    failed_downloads.append({
                        "title": title,
                        "doi": doi,
                        "comments": "Failed to download PDF."
                    })
            else:
                print("Open access PDF not found for this DOI.")
                failed_downloads.append({
                    "title": title,
                    "doi": doi,
                    "comments": "Open access PDF not found for this DOI."
                })
        else:
            failed_downloads.append({
                "title": title,
                "doi": doi,
                "comments": "DOI missing."
            })
        
        pd.DataFrame(failed_downloads).to_csv(failed_downloads_path, index=False)

if __name__ == '__main__':
    papers = DownloadPapers()

    papers.create_directory()
    
    papers.read_and_clean_csvs()
    i = 1
    for file in os.listdir(papers.CLEANED_CSV_DIR):
        df = pd.read_csv(f'{papers.CLEANED_CSV_DIR}/{file}')
        for id, row in df.iterrows():
            doi = row['DOI']
            title = row['Title']
            papers.download_paper_by_doi(doi=doi, title=title, csv_file=file.split('.')[0])
        
        if i == 1:
            break