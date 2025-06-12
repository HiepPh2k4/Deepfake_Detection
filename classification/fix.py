import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score,
    classification_report, average_precision_score, matthews_corrcoef
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

from advanced_transforms import DeepfakeDataset, train_transform, val_test_transform
from advanced_xception import create_optimized_model

# Configuration
DATA_PATH = "G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_full_remote"
MODEL_PATH = "G:/Hiep/Deepfake_Detection/classification/models/model_remote_full_2"
OUTPUT_PATH = "G:/Hiep/Deepfake_Detection/classification/output/output_remote_full_2"
BATCH_SIZE = 32
NUM_WORKERS = 8
PRETRAIN_EPOCHS = 0  # Bỏ qua pretraining vì đã có best_model.pth
FINETUNE_EPOCHS = 2  # Chỉ chạy 2 epoch để tiết kiệm thời gian và tránh overfitting
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
ALPHA = 0.20    # Ưu tiên lớp real (1-alpha=0.80) để giảm nhầm real thành fake
GAMMA = 2.0     # Focal loss parameter để tập trung vào mẫu khó
PATIENCE = 5    # Early stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Định nghĩa SigmoidFocalLoss
class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    """Calculate evaluation metrics."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.5,
        'mcc': matthews_corrcoef(y_true, y_pred),
        'fake_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'real_detection_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'confusion_matrix': cm,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(dataloader, desc='Training'):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return epoch_loss, metrics

def validate_epoch(model, dataloader, criterion, device):
    """Validate one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return epoch_loss, metrics

def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history['train_f1'], label='Train F1')
    axes[1, 0].plot(history['val_f1'], label='Val F1')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history['train_auc'], label='Train AUC')
    axes[1, 1].plot(history['val_auc'], label='Val AUC')
    axes[1, 1].set_title('AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_pred_prob, save_path):
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.5

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_misclassifications(model, dataloader, device, output_path):
    """Analyze misclassified samples."""
    model.eval()
    misclassified = {'fake_as_real': [], 'real_as_fake': []}

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc='Analyzing Errors')):
            images, labels_np = images.to(device), labels.numpy()
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

            for idx in range(len(labels)):
                if labels_np[idx] != preds[idx]:
                    sample_idx = batch_idx * dataloader.batch_size + idx
                    if sample_idx < len(dataloader.dataset):
                        img_path = dataloader.dataset.data.iloc[sample_idx]['image_path']
                        error_info = {
                            'sample_index': sample_idx,
                            'image_path': img_path,
                            'actual_label': 'fake' if labels_np[idx] == 1 else 'real',
                            'predicted_label': 'fake' if preds[idx] == 1 else 'real',
                            'confidence': float(probs[idx]) if preds[idx] == 1 else float(1 - probs[idx])
                        }
                        if labels_np[idx] == 1 and preds[idx] == 0:
                            misclassified['fake_as_real'].append(error_info)
                        else:
                            misclassified['real_as_fake'].append(error_info)

    # Save analysis
    analysis_path = os.path.join(output_path, 'misclassification_analysis.txt')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(
            f"Total misclassified samples: {len(misclassified['fake_as_real']) + len(misclassified['real_as_fake'])}\n")
        f.write(f"Fake misclassified as Real: {len(misclassified['fake_as_real'])}\n")
        f.write(f"Real misclassified as Fake: {len(misclassified['real_as_fake'])}\n\n")

        if misclassified['fake_as_real']:
            f.write("Fake misclassified as Real:\n")
            for i, error in enumerate(
                    sorted(misclassified['fake_as_real'], key=lambda x: x['confidence'], reverse=True)[:10], 1):
                f.write(
                    f"{i}. Index: {error['sample_index']}, Path: {error['image_path']}, Confidence: {error['confidence']:.4f}\n")

        if misclassified['real_as_fake']:
            f.write("\nReal misclassified as Fake:\n")
            for i, error in enumerate(
                    sorted(misclassified['real_as_fake'], key=lambda x: x['confidence'], reverse=True)[:10], 1):
                f.write(
                    f"{i}. Index: {error['sample_index']}, Path: {error['image_path']}, Confidence: {error['confidence']:.4f}\n")

    if misclassified['fake_as_real'] or misclassified['real_as_fake']:
        pd.DataFrame([{'error_type': k, **v} for k, errors in misclassified.items() for v in errors]).to_csv(
            os.path.join(output_path, 'misclassified_samples.csv'), index=False)

    return misclassified

def main():
    """Main function."""
    # Load data
    train_dataset = DeepfakeDataset(f"{DATA_PATH}/train_rgb.csv", transform=train_transform)
    val_dataset = DeepfakeDataset(f"{DATA_PATH}/val_rgb.csv", transform=val_test_transform)
    test_dataset = DeepfakeDataset(f"{DATA_PATH}/test_rgb.csv", transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)

    # Initialize model
    model = create_optimized_model(dropout_rate=0.5).to(DEVICE)

    # Load best model from previous training
    best_model_path = f"{MODEL_PATH}/best_model.pth"
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from: {best_model_path}")
    else:
        raise FileNotFoundError(f"Best model not found at {best_model_path}. Please ensure the file exists.")

    # Initialize criterion and optimizer
    criterion = SigmoidFocalLoss(alpha=ALPHA, gamma=GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
               'train_f1': [], 'val_f1': [], 'train_auc': [], 'val_auc': []}

    # Finetune with early stopping
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = PATIENCE

    for epoch in range(FINETUNE_EPOCHS):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])

        print(
            f"Finetune Epoch {epoch + 1}/{FINETUNE_EPOCHS}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save({'model_state_dict': model.state_dict()}, f"{MODEL_PATH}/best_model.pth")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Lưu mô hình cuối cùng sau khi huấn luyện
    torch.save({'model_state_dict': model.state_dict()}, f"{MODEL_PATH}/final_model.pth")
    print(f"Final model saved to: {MODEL_PATH}/final_model.pth")

    # Test evaluation using final model
    checkpoint = torch.load(f"{MODEL_PATH}/final_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_metrics = validate_epoch(model, test_loader, criterion, DEVICE)

    # Print test results
    print(f"\nTest Results:\nLoss: {test_loss:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, "
          f"F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}, MCC: {test_metrics['mcc']:.4f}")
    print(
        f"Fake Detection: {test_metrics['fake_detection_rate']:.4f}, Real Detection: {test_metrics['real_detection_rate']:.4f}")
    print(f"Confusion Matrix:\nReal: {test_metrics['true_negatives']} TN, {test_metrics['false_positives']} FP\n"
          f"Fake: {test_metrics['false_negatives']} FN, {test_metrics['true_positives']} TP")

    # Classification report
    y_true, y_pred, y_pred_prob = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            y_pred.extend(preds.flatten())
            y_pred_prob.extend(probs.flatten())
            y_true.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

    # Analyze misclassifications
    analyze_misclassifications(model, test_loader, DEVICE, OUTPUT_PATH)

    # Save plots
    plot_training_history(history, f"{OUTPUT_PATH}/training_history.png")
    plot_confusion_matrix(test_metrics['confusion_matrix'], f"{OUTPUT_PATH}/confusion_matrix.png")
    plot_roc_curve(np.array(y_true), np.array(y_pred_prob), f"{OUTPUT_PATH}/roc_curve.png")

    # Save results
    with open(f"{OUTPUT_PATH}/results.txt", 'w') as f:
        f.write(f"Test Results:\nLoss: {test_loss:.4f}\nAccuracy: {test_metrics['accuracy']:.4f}\n"
                f"F1: {test_metrics['f1']:.4f}\nAUC: {test_metrics['auc']:.4f}\nMCC: {test_metrics['mcc']:.4f}\n"
                f"Fake Detection Rate: {test_metrics['fake_detection_rate']:.4f}\n"
                f"Real Detection Rate: {test_metrics['real_detection_rate']:.4f}\n"
                f"Confusion Matrix:\nReal: {test_metrics['true_negatives']} TN, {test_metrics['false_positives']} FP\n"
                f"Fake: {test_metrics['false_negatives']} FN, {test_metrics['true_positives']} TP\n")

    print(f"Results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()