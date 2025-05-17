import requests
import os
import pandas as pd

def get_open_access_pdf_url(doi):
    headers = {
        'Accept': 'application/json'
    }
    email = 'st191452@stud.uni-stuttgart.de'
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        best_oa_location = data.get('best_oa_location')
        if best_oa_location and best_oa_location.get('url_for_pdf'):
            return best_oa_location['url_for_pdf']
    return None

def download_pdf(pdf_url, output_path):
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

def download_paper_by_doi(doi, title, output_dir='papers/hop2'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    failed_downloads_path = 'hop2_hop3_csv/failed_downloads_48-81.csv'

    if os.path.exists(failed_downloads_path):
        failed_downloads = pd.read_csv(failed_downloads_path).to_dict('records')
    else:
        failed_downloads = []
    
    print(f"Looking up DOI: {doi}")
    if doi:
        pdf_url = get_open_access_pdf_url(doi)
        if pdf_url:
            filename = doi.replace('/', '_') + '.pdf'
            output_path = os.path.join(output_dir, filename)
            res = download_pdf(pdf_url, output_path)
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


def get_doi_links():
    df = pd.read_csv('hop2_hop3_csv/hop2.csv')
    for id, row in df.iterrows():
        doi = row['DOI']
        title = row['Title']
        download_paper_by_doi(doi, title)

get_doi_links()