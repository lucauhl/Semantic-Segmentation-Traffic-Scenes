"""
Custom PyTorch Dataset class for Cityscapes semantic segmentation data.

This module implements a PyTorch Dataset wrapper for the Cityscapes dataset, 
enabling efficient data loading and preprocessing with transforms/augmentations for model training.

Reference: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
"""

import torch
import cv2

class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, split="train", image_transform=None, label_transform=None, joint_transform=None):
        """Pairs images with their corresponding labels (labelTrainIds) and appends their file paths in a tuple to data[]"""

        # Validate split parameter to prevent configuration errors
        valid_splits = {"test", "train", "val", "train_extra"}
        if split not in valid_splits:
            raise ValueError(f'Invalid split "{split}". Expected one of {valid_splits}')
        
        self.split = split
        self.img_dir = img_dir / self.split
        self.label_dir = label_dir / self.split
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.joint_transform = joint_transform
        self.data = []

        city_paths = list(self.img_dir.glob("*"))
        for city_path in city_paths:
            city = city_path.name

            # Retrieve all images from city folder
            img_paths = sorted((self.img_dir / city).glob("*_leftImg8bit.png"))

            # Retrieve all labels from city folder
            label_paths = sorted((self.label_dir / city).glob("*_labelTrainIds.png"))

            # Pair each image with its corresponding label
            for img_path, label_path in zip(img_paths, label_paths):
                self.data.append((img_path, label_path))

    def __len__(self):
        """Return total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Load and return an image-label pair at the specified index. Applies transforms/augmentations if set."""

        img_path, label_path = self.data[idx]

        # Load image using OpenCV (default BGR format)
        img = cv2.imread(str(img_path))
        # Convert from BGR to RGB color space for PyTorch compatibility
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load label as grayscale image
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

        # Apply optional transforms/augmentations
        if self.image_transform:
            img = self.image_transform(img)
        if self.label_transform:
            label = self.label_transform(label)
        if self.joint_transform:
            img, label = self.joint_transform(img, label)
        
        return img, label

    
