import boto3
import streamlit as st
import os

BUCKET_NAME = st.secrets["AWS"]["S3_BUCKET_NAME"]
REGION = st.secrets["AWS"]["AWS_REGION"]

FILES = [
    "faiss_index.bin",
    "image_embeddings.npy",
    "image_ids.npy"
]

def download_files():
    print("Downloading models from S3...")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS"]["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS"]["AWS_SECRET_ACCESS_KEY"],
        region_name=REGION
    )

    os.makedirs("embeddings", exist_ok=True)

    for file in FILES:

        local_path = f"embeddings/{file}"

        if not os.path.exists(local_path):

            s3.download_file(
                BUCKET_NAME,
                file,
                local_path
            )