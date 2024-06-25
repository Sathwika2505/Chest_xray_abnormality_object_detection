import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import dill as pickle
from torchvision import transforms

def transform_data():
    def get_train_transform():
        return transforms.Compose([
            transforms.ToTensor()
        ])

    def get_valid_transform():
        return transforms.Compose([
            transforms.ToTensor()
        ])

    class CustomDataset(Dataset):
        def __init__(self, images_path, labels_path, width, height, classes, directory, transforms=None):
            self.transforms = transforms
            self.images_path = images_path
            self.labels_path = labels_path
            self.height = height
            self.width = width
            self.classes = classes
            self.directory = directory
            self.image_file_types = ['*.jpg', '*.jpeg', '*.png']
            self.all_image_paths = []
    
            # Get all the image paths in sorted order
            for file_type in self.image_file_types:
                self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
            self.all_image_paths = sorted(self.all_image_paths)
            self.all_images = [os.path.basename(image_path) for image_path in self.all_image_paths]
    
            print(f"Image path: {self.images_path}")
            print(f"Label path: {self.labels_path}")
            print(f"Number of images: {len(self.all_images)}")
            if len(self.all_images) == 0:
                print(f"No images found in {self.images_path}. Please check the directory.")
    
        def parse_xml_annotation(self, annot_file_path, image_width, image_height):
            boxes = []
            labels = []
            tree = ET.parse(annot_file_path)
            root = tree.getroot()
    
            class_name = root.find('class_name').text
            labels.append(self.classes.index(class_name))
    
            x_min = root.find('x_min').text
            y_min = root.find('y_min').text
            x_max = root.find('x_max').text
            y_max = root.find('y_max').text
    
            xmin = float(x_min) / image_width
            ymin = float(y_min) / image_height
            xmax = float(x_max) / image_width
            ymax = float(y_max) / image_height
    
            if not (np.isnan(xmin) or np.isnan(xmax) or np.isnan(ymin) or np.isnan(ymax)):
                boxes.append([xmin, ymin, xmax, ymax])
    
            return boxes, labels
    
        def parse_txt_annotation(self, annot_file_path, image_width, image_height):
            boxes = []
            labels = []
            with open(annot_file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    class_name = parts[0]
                    x_min, y_min, x_max, y_max = map(float, parts[1:])
    
                    labels.append(self.classes.index(class_name))
    
                    xmin = x_min / image_width
                    ymin = y_min / image_height
                    xmax = x_max / image_width
                    ymax = y_max / image_height
    
                    if not (np.isnan(xmin) or np.isnan(xmax) or np.isnan(ymin) or np.isnan(ymax)):
                        boxes.append([xmin, ymin, xmax, ymax])
    
            return boxes, labels
    
        def __getitem__(self, idx):
            image_name = self.all_images[idx]
            image_path = os.path.join(self.images_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image_resized = cv2.resize(image, (self.width, self.height))
            image_resized /= 255.0
    
            annot_filename = os.path.splitext(image_name)[0]
            annot_file_path_xml = os.path.join(self.labels_path, annot_filename + '.xml')
            annot_file_path_txt = os.path.join(self.labels_path, annot_filename + '.txt')
    
            boxes = []
            labels = []
    
            try:
                if os.path.exists(annot_file_path_xml):
                    boxes, labels = self.parse_xml_annotation(annot_file_path_xml, image.shape[1], image.shape[0])
                elif os.path.exists(annot_file_path_txt):
                    boxes, labels = self.parse_txt_annotation(annot_file_path_txt, image.shape[1], image.shape[0])
                else:
                    raise FileNotFoundError(f"No annotation file found for {image_name}")
            except Exception as e:
                print(f"Error reading annotation file for {image_name}: {e}")
    
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.as_tensor([], dtype=torch.float32)
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd
            image_id = torch.tensor([idx])
            target["image_id"] = image_id
    
            if self.transforms:
                image_resized = self.transforms(image_resized)
    
            if len(target['boxes']) == 0:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
    
            return image_resized, target
    
        def __len__(self):
            return len(self.all_images)

    IMAGE_WIDTH = 800
    IMAGE_HEIGHT = 680
    classes = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'No finding']

    train_dataset = CustomDataset(
        images_path=os.path.join(os.getcwd(), "organized_images/train/images"),
        labels_path=os.path.join(os.getcwd(), "organized_images/train/annotations/xml"),
        labels_txt=os.path.join(os.getcwd(), "organized_images/train/annotations/txt"),
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        classes=classes,
        transforms=get_train_transform(),
        directory = "organized_images"
    )
    print("Train Dataset:", len(train_dataset))

    valid_dataset = CustomDataset(
        images_path=os.path.join(os.getcwd(), "organized_images/test/images"),
        labels_path=os.path.join(os.getcwd(), "organized_images/test/annotations/xml"),
        labels_txt=os.path.join(os.getcwd(), "organized_images/test/annotations/txt"),
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        classes=classes,
        transforms=get_valid_transform(),
        directory = "organized_images"
    )
    print("Validation Dataset:", len(valid_dataset))

    if len(train_dataset) > 0:
        i, a = train_dataset[7]
        print("Sample Image:", i.shape)
        print("Sample Annotations:", a)
    else:
        print("Train dataset is empty.")

    # Save datasets to PKL files
    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('valid_dataset.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)

    return train_dataset

transform_data()
