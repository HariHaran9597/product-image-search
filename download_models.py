import os
import streamlit as st
import urllib.request
import shutil

FILES = [
    "faiss_index.bin",
    "image_embeddings.npy",
    "image_ids.npy"
]

# Configure your public artifact source here (e.g., Hugging Face dataset repo or GitHub Releases)
# Example: "https://huggingface.co/datasets/your-username/your-repo/resolve/main/embeddings"
PUBLIC_BASE_URL = "https://github.com/HariHaran9597/product-image-search/releases/download/v1.0"

def download_from_public(file_name, local_path):
    url = f"{PUBLIC_BASE_URL}/{file_name}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(local_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        return True
    except Exception as e:
        print(f"Public download failed for {file_name}: {e}")
        return False

def download_from_s3(file_name, local_path):
    try:
        import boto3
        if "AWS" not in st.secrets:
            return False
            
        aws_secrets = st.secrets["AWS"]
        if "AWS_ACCESS_KEY_ID" not in aws_secrets or "S3_BUCKET_NAME" not in aws_secrets:
            return False
            
        if aws_secrets["AWS_ACCESS_KEY_ID"] == "optional" or aws_secrets["S3_BUCKET_NAME"] == "optional":
            return False

        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=aws_secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=aws_secrets.get("AWS_REGION", "ap-south-1")
        )
        
        s3.download_file(
            aws_secrets["S3_BUCKET_NAME"],
            file_name,
            local_path
        )
        return True
    except Exception as e:
        print(f"S3 download failed for {file_name}: {e}")
        return False

def download_files():
    os.makedirs("embeddings", exist_ok=True)
    missing_files = []

    for file in FILES:
        local_path = f"embeddings/{file}"
        
        # 1. First check local files
        if os.path.exists(local_path):
            continue
            
        print(f"File {file} not found locally. Attempting to download...")
        
        # 2. Then download from public URLs
        if download_from_public(file, local_path):
            print(f"Successfully downloaded {file} from public URL.")
            continue
            
        print(f"Attempting S3 fallback for {file}...")
        
        # 3. Only use private S3 as optional fallback
        if download_from_s3(file, local_path):
            print(f"Successfully downloaded {file} from S3.")
            continue
            
        # If all fail, mark as missing
        missing_files.append(file)
        
    if missing_files:
        st.error("Artifacts missing. Download failed. Please configure artifact source.")
        st.stop()