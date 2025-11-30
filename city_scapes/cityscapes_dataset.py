import torch
import cv2

class CityScapesDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, split="train", image_transform=None, label_transform=None):
        valid_splits = {"test", "train", "val"}
        if split not in valid_splits:
            raise ValueError(f'Invalid split "{split}". Expected one of {valid_splits}')
        
        self.split = split
        self.img_dir = img_dir / self.split
        self.label_dir = label_dir / self.split
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.data = []

        city_paths = list(self.img_dir.glob("*"))
        for city_path in city_paths:
            city = city_path.name
            # Get all image paths from city folder
            img_paths = sorted((self.img_dir / city).glob("*_leftImg8bit.png"))
            # Get all label paths from city folder
            label_paths = sorted((self.label_dir / city).glob("*_gtFine_labelIds.png"))
            for img_path, label_path in zip(img_paths, label_paths):
                self.data.append((img_path, label_path))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label_path = self.data[idx]

        # Convert opencv default BGR to RGB
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = torch.as_tensor(label, dtype=torch.long)
        return img, label
    
# Reference: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
        

