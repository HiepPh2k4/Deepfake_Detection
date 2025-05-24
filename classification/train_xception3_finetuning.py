import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# Định nghĩa thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Thiết bị: {device}")
if torch.cuda.is_available():
    print(f"Phiên bản PyTorch: {torch.__version__}")
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")


# 1. Tạo Dataset tùy chỉnh
class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        label = self.data.iloc[idx]['label']
        label = 0 if label == 'real' else 1
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, float(label)


# 2. Chuẩn bị transform
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Tải dữ liệu
train_df = pd.read_csv(
    "G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_split_full_remote/train_rgb.csv")
val_df = pd.read_csv("G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_split_full_remote/val_rgb.csv")
test_df = pd.read_csv("G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_split_full_remote/test_rgb.csv")

print(f"Số ảnh huấn luyện: {len(train_df)}")
print(f"Số ảnh kiểm tra: {len(val_df)}")
print(f"Số ảnh thử nghiệm: {len(test_df)}")

# Tạo DataLoader
batch_size = 32
train_dataset = DeepfakeDataset(
    "G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_split_full_remote/train_rgb.csv",
    transform=transform)
val_dataset = DeepfakeDataset(
    "G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_split_full_remote/val_rgb.csv",
    transform=transform)
test_dataset = DeepfakeDataset(
    "G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_split_full_remote/test_rgb.csv",
    transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 4. Tạo mô hình XceptionNet
model = timm.create_model('xception', pretrained=True, num_classes=1).to(device)


# 5. Hàm tính chỉ số
def compute_metrics(y_true, y_pred, y_prob):
    y_pred = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    accuracy = (y_pred == y_true).mean()
    return cm, f1, precision, recall, auc, accuracy


# 6. Hàm huấn luyện
def train_model(model, train_loader, val_loader, epochs, class_weight, save_path, patience=10):
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    best_val_accuracy = -float('inf')
    patience_counter = 0
    history = {'loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [],
               'precision': [], 'val_precision': [], 'recall': [], 'val_recall': [],
               'auc': [], 'val_auc': []}

    for epoch in range(epochs):
        model.train()
        train_loss, train_preds, train_probs, train_labels = 0.0, [], [], []
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Train"):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())
            train_preds.extend((torch.sigmoid(outputs) > 0.5).float().detach().cpu().numpy().flatten())
            train_labels.extend(labels.cpu().numpy().flatten())

        train_loss /= len(train_loader)
        _, _, train_precision, train_recall, train_auc, train_accuracy = compute_metrics(np.array(train_labels),
                                                                                         np.array(train_preds),
                                                                                         np.array(train_probs))

        model.eval()
        val_loss, val_preds, val_probs, val_labels = 0.0, [], [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Val"):
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                val_preds.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())

        val_loss /= len(val_loader)
        _, _, val_precision, val_recall, val_auc, val_accuracy = compute_metrics(np.array(val_labels),
                                                                                 np.array(val_preds),
                                                                                 np.array(val_probs))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        history['auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        print(f"Epoch {epoch + 1}/{epochs}:")
        print(
            f"  Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, AUC: {train_auc:.4f}")
        print(
            f"  Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, AUC: {val_auc:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Đã lưu mô hình tốt nhất tại {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Dừng sớm tại epoch {epoch + 1} do không cải thiện val_accuracy")
            break

    return history


# 7. Pre-training (3 epoch, đóng băng lớp)
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

history_pretrain = train_model(
    model, train_loader, val_loader, epochs=3,
    class_weight={0: 5.0, 1: 1.0},
    save_path="G:/Hiep/Deepfake_Detection/classification/models/models_remote_full/deepfake_model_pretrain_3.pt"
)
print("Đã lưu mô hình sau pre-training!")

# 8. Fine-tuning (15 epoch, mở khóa lớp)
for param in model.parameters():
    param.requires_grad = True

history_finetune = train_model(
    model, train_loader, val_loader, epochs=15,
    class_weight={0: 5.0, 1: 1.0},
    save_path="G:/Hiep/Deepfake_Detection/classification/models/models_remote_full/deepfake_model_best.pt",
    patience=10
)

# 9. Lưu mô hình cuối
torch.save(model.state_dict(),
           "G:/Hiep/Deepfake_Detection/classification/models/models_remote_full/deepfake_model_final_18.pt")
print("Đã lưu mô hình cuối!")

# 10. Tải mô hình tốt nhất để đánh giá
best_model = timm.create_model('xception', pretrained=False, num_classes=1).to(device)
best_model.load_state_dict(
    torch.load("G:/Hiep/Deepfake_Detection/classification/models/models_remote_full/deepfake_model_best.pt"))
best_model.eval()
print("Đã tải mô hình tốt nhất để đánh giá!")

# 11. Đánh giá trên tập test
test_loss, test_preds, test_probs, test_labels = 0.0, [], [], []
criterion = nn.BCEWithLogitsLoss()
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Đánh giá tập test"):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        outputs = best_model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        test_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
        test_preds.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy().flatten())
        test_labels.extend(labels.cpu().numpy().flatten())

test_loss /= len(test_loader)
cm, f1, precision, recall, auc, accuracy = compute_metrics(np.array(test_labels), np.array(test_preds),
                                                           np.array(test_probs))
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Precision: {precision * 100:.2f}%")
print(f"Test Recall: {recall * 100:.2f}%")
print(f"Test AUC: {auc * 100:.2f}%")
print(f"Test F1-score: {f1 * 100:.2f}%")
print(f"Confusion Matrix:\n{cm}")

# 12. Vẽ Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.title("Ma trận nhầm lẫn trên tập Test (Mô hình tốt nhất)")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.savefig("G:/Hiep/Deepfake_Detection/classification/output/remote_full/confusion_matrix_best.png")
plt.show()

# 13. Vẽ lịch sử huấn luyện
history_combined = {}
for key in history_pretrain:
    history_combined[key] = history_pretrain[key] + history_finetune[key]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_combined['loss'], label='Training Loss')
plt.plot(history_combined['val_loss'], label='Validation Loss')
plt.title('Loss qua các Epoch (Pre-Training + Fine-Tuning)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_combined['train_accuracy'], label='Training Accuracy')
plt.plot(history_combined['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy qua các Epoch (Pre-Training + Fine-Tuning)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig("G:/Hiep/Deepfake_Detection/classification/output/remote_full/training_plots_combined.png")
plt.show()