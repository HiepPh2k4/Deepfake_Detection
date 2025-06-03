import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Định nghĩa các tham số
DATA_DIR = "D:/Deepfake_Detection_project/data_preprocessing/output_audio"
MODEL_PATH = "D:/Deepfake_Detection_project/classification/models_voice/models_voice_test/deepfake_audio_model.pth"
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Chuẩn bị dữ liệu
def get_data_loaders():
    # Biến đổi cho tập huấn luyện (có tăng cường dữ liệu)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(299, padding=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Biến đổi cho tập test và validation (không tăng cường dữ liệu)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Tải tập dữ liệu
    train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=train_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "test"), transform=test_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "validation"), transform=test_transform)

    # Tạo data loader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Kiểm tra số lượng mẫu
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, test_loader, val_loader, train_dataset.classes

# 2. Xây dựng mô hình
def build_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

# 3. Huấn luyện mô hình
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_auc = 0.0
    train_losses = []
    val_losses = []
    val_aucs = []

    for epoch in range(num_epochs):
        # Huấn luyện
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Đánh giá trên tập validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_auc = roc_auc_score(all_labels, all_preds)
        val_aucs.append(val_auc)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Lưu mô hình tốt nhất
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model with Val AUC: {best_val_auc:.4f}")

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.close()

    return model

# 4. Đánh giá mô hình
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # Tính các chỉ số
    auc = roc_auc_score(all_labels, all_preds)
    predictions = [1 if p > 0.5 else 0 for p in all_preds]
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, predictions, average="binary")
    accuracy = sum([1 if p == l else 0 for p, l in zip(predictions, all_labels)]) / len(all_labels)

    print("\nTest Results:")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return auc, accuracy, precision, recall, f1

# 5. Hàm chính
def main():
    # Tải dữ liệu
    train_loader, test_loader, val_loader, classes = get_data_loaders()
    print(f"Classes: {classes}")

    # Xây dựng mô hình
    model = build_model(num_classes=len(classes))

    # Xử lý dữ liệu không cân bằng
    # Đếm số mẫu mỗi lớp trong tập train
    train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"))
    class_counts = torch.bincount(torch.tensor(train_dataset.targets))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # Tối ưu hóa
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Huấn luyện
    model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)

    # Đánh giá
    auc, accuracy, precision, recall, f1 = evaluate_model(model, test_loader)

    # Lưu kết quả
    with open("evaluation_results.txt", "w") as f:
        f.write(f"Test AUC: {auc:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall: {recall:.4f}\n")
        f.write(f"Test F1-Score: {f1:.4f}\n")

if __name__ == "__main__":
    main()