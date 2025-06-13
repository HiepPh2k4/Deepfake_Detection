import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from advanced_transforms import DeepfakeDataset, train_transform, val_test_transform
from train_utils import train_and_evaluate, FocalLoss
import os

if __name__ == '__main__':
    # Set parameters
    DATA_PATH = "G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_test_remote"
    MODEL_PATH = "G:/Hiep/Deepfake_Detection/classification/models/model_densenet121"
    OUTPUT_PATH = "G:/Hiep/Deepfake_Detection/classification/output/output_densenet121"
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4
    ALPHA = 0.80  # Weight for REAL (0), so 1 - ALPHA = 0.20 for FAKE (1)
    GAMMA = 2.0
    PATIENCE = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRETRAIN_EPOCHS = 3
    FINETUNE_EPOCHS = 15

    # Load data
    train_csv = os.path.join(DATA_PATH, "train.csv")
    val_csv = os.path.join(DATA_PATH, "val.csv")
    test_csv = os.path.join(DATA_PATH, "test.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Number of training images: {len(train_df)}")
    print(f"Number of validation images: {len(val_df)}")
    print(f"Number of test images: {len(test_df)}")

    # Prepare DenseNet-121 model
    model = models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 1)
    model = model.to(DEVICE)

    # Create DataLoaders
    train_dataset = DeepfakeDataset(train_csv, transform=train_transform)
    val_dataset = DeepfakeDataset(val_csv, transform=val_test_transform)
    test_dataset = DeepfakeDataset(test_csv, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Pretrain (freeze backbone)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    criterion = FocalLoss(alpha=torch.tensor([ALPHA, 1 - ALPHA]), gamma=GAMMA)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE)

    print("Starting pretraining...")
    model, metrics, train_losses, val_losses, train_accs, val_accs = train_and_evaluate(
        model, train_loader, val_loader, test_loader, criterion, optimizer,
        model_name="DenseNet121", num_epochs=PRETRAIN_EPOCHS, device=DEVICE, save_dir=OUTPUT_PATH, patience=PATIENCE
    )

    # Finetune (unfreeze all)
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE)

    print("Starting finetuning...")
    model, metrics, train_losses, val_losses, train_accs, val_accs = train_and_evaluate(
        model, train_loader, val_loader, test_loader, criterion, optimizer,
        model_name="DenseNet121", num_epochs=FINETUNE_EPOCHS, device=DEVICE, save_dir=OUTPUT_PATH, patience=PATIENCE
    )

    # Save final model
    os.makedirs(MODEL_PATH, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(MODEL_PATH, "densenet121_deepfake.pth"))
    print("Saved model DenseNet121!")