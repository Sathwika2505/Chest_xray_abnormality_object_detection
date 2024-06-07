import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET
import dill as pickle

def transform_data():
    def get_train_aug():
        return A.Compose([
            A.MotionBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, p=0.5),
            A.ColorJitter(p=0.5),
            A.RandomGamma(p=0.2),
            A.RandomFog(p=0.2),
            ToTensorV2(p=1.0),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def get_train_transform():
        return A.Compose([
            ToTensorV2(p=1.0),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def get_valid_transform():
        return A.Compose([
            ToTensorV2(p=1.0),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    class CustomDataset(Dataset):
        def __init__(self, images_path, labels_path, width, height, classes, transforms=None, use_train_aug=False, train=False, mosaic=False):
            self.transforms = transforms
            self.use_train_aug = use_train_aug
            self.images_path = images_path
            self.labels_path = labels_path
            self.height = height
            self.width = width
            self.classes = classes
            self.train = train
            self.mosaic = mosaic
            self.image_file_types = ['*.jpg', '*.jpeg', '*.png']
            self.all_image_paths = []

            # get all the image paths in sorted order
            for file_type in self.image_file_types:
                self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
            self.all_image_paths = sorted(self.all_image_paths)
            self.all_images = [os.path.basename(image_path) for image_path in self.all_image_paths]

            print(f"Image path: {self.images_path}")
            print(f"Label path: {self.labels_path}")
            print(f"Number of images: {len(self.all_images)}")
            if len(self.all_images) == 0:
                print(f"No images found in {self.images_path}. Please check the directory.")

        def load_image_and_labels(self, index):
            if index >= len(self.all_images):
                raise IndexError("Index out of range")
            image_name = self.all_images[index]
            image_path = os.path.join(self.images_path, image_name)

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading image {image_path}. Please check if the file is corrupted.")
                return None, None, [], [], [], [], (0, 0)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image_resized = cv2.resize(image, (self.width, self.height))
            image_resized /= 255.0

            # Capture the corresponding XML file for getting the annotations
            annot_filename = image_name[:-4] + '.xml'
            annot_file_path = os.path.join(self.labels_path, annot_filename)

            if not os.path.exists(annot_file_path):
                print(f"Annotation file {annot_file_path} does not exist")
                return image, image_resized, [], [], [], [], (0, 0)

            boxes = []
            labels = []
            tree = ET.parse(annot_file_path)
            root = tree.getroot()

            # Get the height and width of the image
            image_width = image.shape[1]
            image_height = image.shape[0]

            # Box coordinates for XML files are extracted and corrected for image size given
            for member in root.findall('object'):
                labels.append(self.classes.index(member.find('name').text))
                bndbox = member.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # Resize the bounding boxes
                xmin_final = (xmin / image_width) * self.width
                xmax_final = (xmax / image_width) * self.width
                ymin_final = (ymin / image_height) * self.height
                ymax_final = (ymax / image_height) * self.height

                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

            # Bounding box to tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Area of the bounding boxes
            if len(boxes.size()) == 1:
                area = torch.tensor([0.0], dtype=torch.float32)
            else:
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # No crowd instances
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            # Labels to tensor
            labels = torch.as_tensor(labels, dtype=torch.int64)
            return image, image_resized, boxes, labels, area, iscrowd, (image_width, image_height)

        def __getitem__(self, idx):
            image, image_resized, boxes, labels, area, iscrowd, dims = self.load_image_and_labels(index=idx)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["image_id"] = torch.tensor([idx])

            if self.use_train_aug:
                train_aug = get_train_aug()
                sample = train_aug(image=image_resized, bboxes=target['boxes'], labels=labels)
                image_resized = sample['image']
                target['boxes'] = torch.Tensor(sample['bboxes'])
            else:
                sample = self.transforms(image=image_resized, bboxes=target['boxes'], labels=labels)
                image_resized = sample['image']
                target['boxes'] = torch.Tensor(sample['bboxes'])

            return image_resized, target

        def __len__(self):
            return len(self.all_images)

    IMAGE_WIDTH = 800
    IMAGE_HEIGHT = 680
    classes = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']

    train_dataset = CustomDataset(
        images_path=os.path.join(os.getcwd(), "organized_images/train/images"),
        labels_path=os.path.join(os.getcwd(), "organized_images/train/annotations"),
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        classes=classes,
        transforms=get_train_transform() 
    )
    print("Train Dataset:", len(train_dataset))

    valid_dataset = CustomDataset(
        images_path=os.path.join(os.getcwd(), "organized_images/test/images"),
        labels_path=os.path.join(os.getcwd(), "organized_images/test/annotations"),
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        classes=classes,
        transforms=get_valid_transform()
    )
    print("Validation Dataset:", len(valid_dataset))

    if len(train_dataset) > 0:
        print(train_dataset)
        i, a = train_dataset[2]
        print("Sample Image:", i)
        print("Sample Annotations:", a)
    else:
        print("Train dataset is empty.")

    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('valid_dataset.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)

    return train_dataset

transform_data()
