import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float() / 255.0
    return img

class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label_str = self.df.iloc[idx]['label']
        if pd.isna(label_str) or label_str not in ["real", "fake"]:
            raise ValueError(f"Invalid label '{label_str}' for image: {img_path}")
        label = 0 if label_str == "real" else 1  # 0 for REAL, 1 for FAKE
        image = load_and_preprocess_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32), img_path

def data_generator(csv_file, batch_size, transform=None):
    dataset = DeepfakeDataset(csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    return dataloader