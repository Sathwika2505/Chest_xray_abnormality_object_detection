import os
import pandas as pd
import boto3
from io import BytesIO
from PIL import Image
import random
import shutil
import pathlib
import xml.etree.ElementTree as ET
from data_extraction import extract_data
from sklearn.model_selection import train_test_split
import io 

def read_csv_from_s3(bucket_name, csv_file_key):
    try:
        print("Accessing CSV file from S3")
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket_name, Key=csv_file_key)
        data = response['Body'].read()
        df = pd.read_csv(io.BytesIO(data))
        print("CSV file loaded into DataFrame")
        print(f"Total number of entries in CSV file: {df['image_id'].nunique()}")
        return df
    except Exception as e:
        print(f"Error downloading CSV file from S3: {e}")
        return None

def convert_row_to_xml(row, output_dir):
    root = ET.Element("annotation")
    for col in row.index:
        child = ET.SubElement(root, col)
        child.text = str(row[col])
    tree = ET.ElementTree(root)
    xml_file_path = os.path.join(output_dir, f"{row['image_id']}.xml")
    tree.write(xml_file_path)

def organize_images_and_annotations(df, images_dir, images_output_dir, annotations_output_dir):
    if df is None:
        print("DataFrame is None. Exiting the function.")
        return
    
    saved_files = []
    missing_files = []
    
    for index, row in df.iterrows():
        image_id = row['image_id']
        image_file = f"{image_id}.jpg"
        image_path = os.path.join(images_dir, image_file)
        
        if os.path.exists(image_path):
            # Ensure output directories exist
            pathlib.Path(images_output_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(annotations_output_dir).mkdir(parents=True, exist_ok=True)
            
            # Copy the image
            dest_image_path = os.path.join(images_output_dir, image_file)
            shutil.copy(image_path, dest_image_path)
            saved_files.append(dest_image_path)
            
            # Create the corresponding XML file in the annotations directory
            convert_row_to_xml(row, annotations_output_dir)
        else:
            print(f"Image {image_file} not found in {images_dir}.")
            missing_files.append(image_file)
    
    print("Images and annotations saved successfully.")
    print(f"Total images missing: {len(missing_files)}")
    return saved_files, missing_files

def open_random_image(path):
    try:
        all_files = os.listdir(path)
        random_image_file = random.choice(all_files)
        image_path = os.path.join(path, random_image_file)
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error opening image from {path}: {e}")
        return None

def save_random_images_from_each_class(base_dir, class_list):
    for class_name in class_list:
        random_image = open_random_image(base_dir)
        if random_image:
            filename = f"{class_name.replace(' ', '')}.jpg"
            random_image.save(filename)
        else:
            print(f"No image saved for class {class_name}")

def split_data(df, images_dir, output_dir, test_size=0.2):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['class_name'], random_state=42)
    
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_annotations_dir = os.path.join(output_dir, 'train', 'annotations')
    test_images_dir = os.path.join(output_dir, 'test', 'images')
    test_annotations_dir = os.path.join(output_dir, 'test', 'annotations')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_annotations_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_annotations_dir, exist_ok=True)
    
    organize_images_and_annotations(train_df, images_dir, train_images_dir, train_annotations_dir)
    organize_images_and_annotations(test_df, images_dir, test_images_dir, test_annotations_dir)
    
    train_df.to_csv(os.path.join(output_dir, 'train', 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test', 'test.csv'), index=False)
    
    print(f"Data split into training and test sets. Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, test_df

def main():
    bucket_name = 'deeplearning-mlops-demo'
    csv_file_key = 'train.csv'
    local_dir = "extracted_images/train"
    output_dir = 'organized_images'
    class_list = [
        "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
        "Consolidation", "ILD", "Infiltration", "Lung Opacity", "No finding",
        "Nodule", "Other lesion", "Pleural effusion", "Pleural thickening",
        "Pneumothorax", "Pulmonary fibrosis"
    ]

    # Step 1: Download and extract zip file
    zip_file = extract_data()

    # Step 2: Read the CSV file
    csv_df = read_csv_from_s3(bucket_name, csv_file_key)

    if csv_df is not None:
        # Step 3: Split data into training and test sets
        train_df, test_df = split_data(csv_df, local_dir, output_dir)

        # Step 4: Save a random image from each class for the training set
        save_random_images_from_each_class(os.path.join(output_dir, 'train', 'images'), class_list)

if __name__ == "__main__":
    main()
