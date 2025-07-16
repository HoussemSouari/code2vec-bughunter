import os
import requests
import tarfile
import logging

logging.basicConfig(level=logging.INFO)

# URL for the CodeSearchNet Python dataset
DATASET_URL = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip"
ZIP_FILE = "python.zip"
EXTRACT_DIR = "codesearchnet_python"

def download_file(url, filename):
    """Download a file from a URL."""
    if os.path.exists(filename):
        logging.info(f"File {filename} already exists. Skipping download.")
        return
    logging.info(f"Downloading {url} to {filename}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    logging.info("Download complete.")

def extract_zip(filename, extract_dir):
    """Extract a zip file."""
    if os.path.exists(extract_dir):
        logging.info(f"Directory {extract_dir} already exists. Skipping extraction.")
        return
    logging.info(f"Extracting {filename} to {extract_dir}...")
    import zipfile
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    logging.info("Extraction complete.")

if __name__ == "__main__":
    download_file(DATASET_URL, ZIP_FILE)
    extract_zip(ZIP_FILE, EXTRACT_DIR)
