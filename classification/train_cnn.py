# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# from tqdm import tqdm
# import time
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# class DeepfakeDataset(Dataset):
#     """Dataset class tối ưu cho deepfake detection"""
#
#     def __init__(self, image_paths, labels, transform=None, augment_fake=True):
#         self.image_paths = image_paths
#         self.labels = labels
#         self.transform = transform
#         self.augment_fake = augment_fake
#
#         # Heavy augmentation cho fake images để balance dataset
#         self.heavy_transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.RandomHorizontalFlip(p=0.7),
#             transforms.RandomRotation(degrees=20),
#             transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
#             transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
#             transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))
#         ])
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         label = self.labels[idx]
#
#         try:
#             image = Image.open(image_path).convert('RGB')
#
#             # Apply heavier augmentation to fake images during training
#             if self.augment_fake and label == 1 and np.random.random() > 0.3:
#                 image = self.heavy_transform(image)
#             elif self.transform:
#                 image = self.transform(image)
#
#             return image, torch.tensor(label, dtype=torch.float32)
#
#         except Exception as e:
#             print(f"Error loading {image_path}: {e}")
#             # Return dummy data
#             dummy_image = torch.zeros(3, 224, 224)
#             return dummy_image, torch.tensor(label, dtype=torch.float32)
#
#
# def get_custom_transforms():
#     """Custom transforms cho deepfake detection"""
#
#     # Training transforms với focus vào artifacts
#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation(degrees=10),
#
#         # Color augmentation để robust với compression artifacts
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
#
#         # Geometric transforms
#         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
#
#         # Blur để simulate compression
#         transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
#
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#
#         # Random erasing để tăng robustness
#         transforms.RandomErasing(p=0.1, scale=(0.02, 0.2))
#     ])
#
#     # Validation/Test transforms
#     val_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     return train_transform, val_transform
#
#
# def load_faceforensics_data(data_dir, val_split=0.2, test_split=0.1):
#     """Load FaceForensics++ dataset"""
#
#     image_paths = []
#     labels = []
#
#     # Scan real images
#     real_dir = os.path.join(data_dir, 'real')
#     if os.path.exists(real_dir):
#         for img_name in os.listdir(real_dir):
#             if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 image_paths.append(os.path.join(real_dir, img_name))
#                 labels.append(0)  # Real = 0
#
#     # Scan fake images
#     fake_dir = os.path.join(data_dir, 'fake')
#     if os.path.exists(fake_dir):
#         for img_name in os.listdir(fake_dir):
#             if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 image_paths.append(os.path.join(fake_dir, img_name))
#                 labels.append(1)  # Fake = 1
#
#     print(f"Total images: {len(image_paths)}")
#     print(f"Real: {labels.count(0)}, Fake: {labels.count(1)}")
#
#     # Split data
#     train_paths, temp_paths, train_labels, temp_labels = train_test_split(
#         image_paths, labels, test_size=(val_split + test_split),
#         random_state=42, stratify=labels
#     )
#
#     val_paths, test_paths, val_labels, test_labels = train_test_split(
#         temp_paths, temp_labels, test_size=test_split / (val_split + test_split),
#         random_state=42, stratify=temp_labels
#     )
#
#     return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
#
#
# def create_data_loaders(data_splits, batch_size=32):
#     """Create data loaders"""
#
#     (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = data_splits
#     train_transform, val_transform = get_custom_transforms()
#
#     # Create datasets
#     train_dataset = DeepfakeDataset(train_paths, train_labels, train_transform, augment_fake=True)
#     val_dataset = DeepfakeDataset(val_paths, val_labels, val_transform, augment_fake=False)
#     test_dataset = DeepfakeDataset(test_paths, test_labels, val_transform, augment_fake=False)
#
#     # Create data loaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                               num_workers=4, pin_memory=True, drop_last=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                             num_workers=4, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
#                              num_workers=4, pin_memory=True)
#
#     print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
#
#     return train_loader, val_loader, test_loader
#
#
# class DeepfakeTrainer:
#     """Custom trainer cho deepfake detection"""
#
#     def __init__(self, model, device, model_type='basic'):
#         self.model = model.to(device)
#         self.device = device
#         self.model_type = model_type
#
#         # Optimizer với different learning rates
#         if model_type == 'enhanced':
#             # Different learning rates cho different parts
#             classifier_params = list(self.model.classifier.parameters())
#             aux_params = list(self.model.aux_classifier.parameters())
#             backbone_params = [p for n, p in self.model.named_parameters()
#                                if not any(x in n for x in ['classifier', 'aux_classifier'])]
#
#             self.optimizer = optim.AdamW([
#                 {'params': backbone_params, 'lr': 1e-4},
#                 {'params': classifier_params, 'lr': 2e-4},
#                 {'params': aux_params, 'lr': 2e-4}
#             ], weight_decay=1e-4)
#         else:
#             self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
#
#         # Learning rate scheduler
#         self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
#         )
#
#         # Loss functions
#         if model_type == 'enhanced':
#             self.criterion =