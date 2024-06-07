import boto3 
import os
import zipfile
from io import BytesIO
import io

def extract_data():

    s3 = boto3.client('s3')
    bucket_name = 'deeplearning-mlops-demo'
    file_key = 'trainimages.zip'
    
    with BytesIO() as zip_buffer:
        s3.download_fileobj(bucket_name, file_key, zip_buffer)
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            zip_ref.extractall("extracted_images")
            print(zip_ref)
    return zip_ref

extract_data()
