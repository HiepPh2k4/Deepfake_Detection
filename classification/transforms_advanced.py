import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import random
import cv2


class RandomJPEGCompression:
    """Simulate JPEG compression artifacts"""

    def __init__(self, quality_range=(30, 95)):
        self.quality_range = quality_range

    def __call__(self, img):
        if random.random() > 0.5:
            quality = random.randint(*self.quality_range)
            img_np = np.array(img)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', img_np, encode_param)
            decimg = cv2.imdecode(encimg, 1)
            return Image.fromarray(decimg)
        return img


class RandomBlur:
    """Apply random blur to simulate video compression"""

    def __init__(self, kernel_range=(3, 7)):
        self.kernel_range = kernel_range

    def __call__(self, img):
        if random.random() > 0.5:
            kernel_size = random.choice(range(self.kernel_range[0], self.kernel_range[1] + 1, 2))
            return TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size])
        return img


class RandomDownScale:
    """Downscale and upscale to simulate low quality"""

    def __init__(self, scale_range=(0.7, 0.95)):
        self.scale_range = scale_range

    def __call__(self, img):
        if random.random() > 0.5:
            scale = random.uniform(*self.scale_range)
            w, h = img.size
            new_w, new_h = int(w * scale), int(h * scale)
            img = TF.resize(img, (new_h, new_w), interpolation=Image.BILINEAR)
            img = TF.resize(img, (h, w), interpolation=Image.BILINEAR)
        return img


class ColorJitterPro:
    """Enhanced color jitter for deepfake detection"""

    def __init__(self):
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        )

    def __call__(self, img):
        # Standard color jitter
        img = self.color_jitter(img)

        # Additional color manipulations
        if random.random() > 0.7:
            # Random channel shuffle
            img_np = np.array(img)
            channels = [0, 1, 2]
            random.shuffle(channels)
            img_np = img_np[:, :, channels]
            img = Image.fromarray(img_np)

        return img


class GridMask:
    """GridMask augmentation for better generalization"""

    def __init__(self, d_range=(30, 60), ratio=0.5, prob=0.5):
        self.d_range = d_range
        self.ratio = ratio
        self.prob = prob

    def __call__(self, img):
        if random.random() > self.prob:
            return img

        h, w = img.size[1], img.size[0]
        d = random.randint(*self.d_range)

        # Create grid mask
        mask = np.ones((h, w), dtype=np.float32)
        st_h = random.randint(0, d)
        st_w = random.randint(0, d)

        for i in range(st_h, h, d):
            for j in range(st_w, w, d):
                mask[i:i + int(d * self.ratio), j:j + int(d * self.ratio)] = 0

        # Apply mask
        img_np = np.array(img).astype(np.float32)
        img_np = img_np * mask[:, :, np.newaxis]

        return Image.fromarray(img_np.astype(np.uint8))


class CutMix:
    """CutMix augmentation - to be applied during training"""

    def __init__(self, prob=0.5, beta=1.0):
        self.prob = prob
        self.beta = beta

    def __call__(self, img):
        # This is a placeholder - actual CutMix is applied in training loop
        return img


# Strong augmentation for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),

    # Deepfake-specific augmentations
    RandomJPEGCompression(quality_range=(40, 95)),
    RandomBlur(kernel_range=(3, 7)),
    RandomDownScale(scale_range=(0.75, 0.95)),

    # Advanced augmentations
    GridMask(d_range=(30, 60), ratio=0.5, prob=0.3),
    ColorJitterPro(),

    # Random rotation
    transforms.RandomRotation(degrees=10),

    # Random affine
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=5
    ),

    # Standard augmentations
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomPosterize(bits=2, p=0.1),
    transforms.RandomSolarize(threshold=128, p=0.1),

    # Normalize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # Random erasing (similar to cutout)
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3))
])

# Validation/Test transform
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# TTA (Test Time Augmentation) transforms
tta_transforms = [
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.functional.hflip,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
]


class DeepfakeDataset(Dataset):
    """Dataset class for deepfake detection with advanced augmentation support"""

    def __init__(self, csv_path, transform=None, use_soft_labels=False):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.use_soft_labels = use_soft_labels

        # Convert text labels to numeric if needed
        if self.data['label'].dtype == 'object':
            # Map text labels to numeric
            label_map = {'real': 0, 'fake': 1, 'Real': 0, 'Fake': 1, 'REAL': 0, 'FAKE': 1}
            self.data['label'] = self.data['label'].map(label_map)
            if self.data['label'].isna().any():
                # If mapping failed, try numeric conversion
                self.data = pd.read_csv(csv_path)
                self.data['label'] = pd.to_numeric(self.data['label'], errors='coerce')

        # Calculate class weights for balanced sampling
        self.labels = self.data['label'].values
        self.class_weights = self._calculate_class_weights()

    def _calculate_class_weights(self):
        """Calculate class weights for balanced sampling"""
        unique, counts = np.unique(self.labels, return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return {int(unique[i]): weights[i] for i in range(len(unique))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        label = row['label']

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        # Soft labels for label smoothing (optional)
        if self.use_soft_labels:
            if label == 0:
                label = np.random.uniform(0.0, 0.1)  # Real: 0.0-0.1
            else:
                label = np.random.uniform(0.9, 1.0)  # Fake: 0.9-1.0

        return image, label

    def get_sample_weights(self):
        """Get sample weights for balanced sampling"""
        weights = [self.class_weights[label] for label in self.labels]
        return weights


# Additional utility functions
def get_train_transform_strong():
    """Get very strong augmentation for hard cases"""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),

        # Very strong augmentations
        RandomJPEGCompression(quality_range=(30, 80)),
        RandomBlur(kernel_range=(3, 9)),
        RandomDownScale(scale_range=(0.6, 0.9)),

        GridMask(d_range=(20, 70), ratio=0.6, prob=0.5),
        ColorJitterPro(),

        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=10
        ),

        transforms.RandomGrayscale(p=0.2),
        transforms.RandomPosterize(bits=2, p=0.2),
        transforms.RandomSolarize(threshold=128, p=0.2),
        transforms.RandomInvert(p=0.05),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3))
    ])


def get_progressive_transform(epoch, max_epochs):
    """Get progressively stronger augmentation as training progresses"""
    strength = min(1.0, epoch / (max_epochs * 0.7))  # Max strength at 70% of training

    jpeg_quality = (int(70 - 30 * strength), 95)
    blur_kernel = (3, int(3 + 4 * strength))
    downscale = (0.9 - 0.15 * strength, 0.95)

    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9 - 0.1 * strength, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),

        RandomJPEGCompression(quality_range=jpeg_quality),
        RandomBlur(kernel_range=blur_kernel),
        RandomDownScale(scale_range=downscale),

        GridMask(d_range=(30, 60), ratio=0.5, prob=0.3 * strength),
        ColorJitterPro() if strength > 0.3 else transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),

        transforms.RandomRotation(degrees=int(10 * strength)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3 * strength, scale=(0.02, 0.15), ratio=(0.3, 3.3))
    ])