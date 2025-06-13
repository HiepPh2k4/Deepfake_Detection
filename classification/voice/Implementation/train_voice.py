import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from crnn_model import CRNN
import os

# Constants
DATA_DIR = "/data_preprocessing/output_audio"
MODEL_PATH = "/classification/models/ver1/ver1/deepfake_audio_model.pth"
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(299, padding=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def build_model(num_classes):
    model = CRNN(num_classes)
    return model.to(DEVICE)


def train_and_evaluate():
    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "validation"), transform=test_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Compute class weights
    class_counts = np.bincount([y for _, y in train_dataset.samples])
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)

    # Build and train model
    model = build_model(len(train_dataset.classes))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    best_val_auc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)
        train_acc = accuracy_score(train_labels, [1 if p > 0.5 else 0 for p in train_preds])

        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_preds])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, Train Acc={train_acc:.4f}")
        print(f"           Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}, Val Acc={val_acc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), MODEL_PATH)

    # Test evaluation
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            test_preds.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, [1 if p > 0.5 else 0 for p in test_preds])
    test_precision = precision_score(test_labels, [1 if p > 0.5 else 0 for p in test_preds])
    test_recall = recall_score(test_labels, [1 if p > 0.5 else 0 for p in test_preds])
    test_f1 = f1_score(test_labels, [1 if p > 0.5 else 0 for p in test_preds])

    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")

    # Save results
    with open("../output/ver1/evaluation_results.txt", "w") as f:
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write(f"Test F1-Score: {test_f1:.4f}\n")

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_aucs, label="Train AUC")
    plt.plot(val_aucs, label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.savefig("training_plot.png")
    plt.close()


if __name__ == "__main__":
    train_and_evaluate()