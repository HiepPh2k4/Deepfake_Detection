import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             average_precision_score, precision_score, recall_score)
import json
import os
import time
from pathlib import Path
from tqdm import tqdm
import random
import warnings

warnings.filterwarnings('ignore')

from advanced_xception import ImprovedXception
from Deepfake_Detection.classification.face.Implementation.advanced_transforms import DeepfakeDataset, train_transform, val_test_transform

# Configuration
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

DATA_PATH = "G:/Hiep/Deepfake_Detection/data_preprocessing/output_split"
MODEL_PATH = "G:/Hiep/Deepfake_Detection/classification/face/models/enhanced_xception"
OUTPUT_PATH = "G:/Hiep/Deepfake_Detection/classification/face/output/enhanced_xception"
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 2e-4
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
ALPHA = 0.82
GAMMA = 2.0
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAIN_EPOCHS = 3
FINETUNE_EPOCHS = 15
MIXUP_ALPHA = 0.0  # Disabled to match baseline
SAVE_ATTENTION_FREQ = 1  # Save attention maps every epoch
NUM_ATTENTION_SAMPLES = 5  # Save 5 samples per dataset per epoch

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
ATTENTION_DIR = os.path.join(OUTPUT_PATH, "attention_maps")
os.makedirs(os.path.join(ATTENTION_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(ATTENTION_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(ATTENTION_DIR, "test"), exist_ok=True)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.82, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).to(DEVICE)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        alpha_t = self.alpha[0] * (1 - targets) + self.alpha[1] * targets
        focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Metrics calculation
def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
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

# Attention visualization
def visualize_attention_map(model, image, label, pred_prob, save_path, image_name=""):
    model.eval()
    with torch.no_grad():
        image_input = image.unsqueeze(0).to(DEVICE)
        attention_maps = model.get_attention_maps(image_input)

    image_np = image.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np * std + mean).clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{image_name}\nTrue: {"Fake" if label == 1 else "Real"} | '
                 f'Pred: {"Fake" if pred_prob > 0.5 else "Real"} ({pred_prob:.3f})',
                 fontsize=12)

    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    att_map = attention_maps['cbam'][0, 0].cpu().numpy()
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
    axes[1].imshow(image_np)
    im = axes[1].imshow(att_map, cmap='Reds', alpha=0.5)
    axes[1].set_title('CBAM Attention Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    att_map_self = attention_maps['self_attention'][0, 0].cpu().numpy()
    att_map_self = (att_map_self - att_map_self.min()) / (att_map_self.max() - att_map_self.min() + 1e-8)
    axes[2].imshow(image_np)
    im = axes[2].imshow(att_map_self, cmap='Blues', alpha=0.5)
    axes[2].set_title('Self-Attention Map')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_attention_samples(model, dataloader, epoch, dataset_type, num_samples=5):
    model.eval()
    save_dir = os.path.join(ATTENTION_DIR, dataset_type, f"epoch_{epoch:03d}")
    os.makedirs(save_dir, exist_ok=True)

    indices = random.sample(range(len(dataloader.dataset)), min(num_samples, len(dataloader.dataset)))
    for idx in indices:
        image, label, path = dataloader.dataset[idx]
        with torch.no_grad():
            image_input = image.unsqueeze(0).to(DEVICE)
            output = model(image_input)
            pred_prob = torch.sigmoid(output).item()
        image_name = Path(path).name
        save_path = os.path.join(save_dir, f"{image_name}_true{'fake' if label == 1 else 'real'}_pred{pred_prob:.3f}.png")
        visualize_attention_map(model, image, label, pred_prob, save_path, image_name)

def train_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    progress_bar = tqdm(dataloader, desc='Training')
    for images, labels, _ in progress_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()

        with autocast(enabled=DEVICE.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

        current_loss = running_loss / (len(all_preds) * images.size(0) / len(images))
        progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return epoch_loss, metrics

def validate_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    progress_bar = tqdm(dataloader, desc='Validation')
    with torch.no_grad():
        for images, labels, _ in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            with autocast(enabled=DEVICE.type == 'cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

            current_loss = running_loss / (len(all_preds) * images.size(0) / len(images))
            progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return epoch_loss, metrics

def save_plots(history, test_metrics, y_true, y_pred_prob):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes[0, 0].plot(history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_acc'], 'b-', label='Train')
    axes[0, 1].plot(history['val_acc'], 'r-', label='Val')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[0, 2].plot(history['train_f1'], 'b-', label='Train')
    axes[0, 2].plot(history['val_f1'], 'r-', label='Val')
    axes[0, 2].set_title('F1 Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    axes[1, 0].plot(history['train_auc'], 'b-', label='Train')
    axes[1, 0].plot(history['val_auc'], 'r-', label='Val')
    axes[1, 0].set_title('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history['train_precision'], 'b-', label='Train')
    axes[1, 1].plot(history['val_precision'], 'r-', label='Val')
    axes[1, 1].set_title('Precision')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    axes[1, 2].plot(history['train_recall'], 'b-', label='Train')
    axes[1, 2].plot(history['val_recall'], 'r-', label='Val')
    axes[1, 2].set_title('Recall')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.suptitle('Training Progress - ImprovedXception')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "training_history.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', lw=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Test Set')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    ap = average_precision_score(y_true, y_pred_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f'AP = {ap:.4f}', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Test Set')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "precision_recall_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_misclassifications(model, test_loader):
    model.eval()
    misclassified = []

    progress_bar = tqdm(test_loader, desc='Analyzing misclassifications')
    with torch.no_grad():
        for images, labels, paths in progress_bar:
            images = images.to(DEVICE)
            labels_np = labels.numpy()
            with autocast(enabled=DEVICE.type == 'cuda'):
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

            for idx in range(len(labels)):
                if labels_np[idx] != preds[idx]:
                    misclassified.append({
                        'image_path': paths[idx],
                        'true_label': 'fake' if labels_np[idx] == 1 else 'real',
                        'predicted_label': 'fake' if preds[idx] == 1 else 'real',
                        'confidence': float(probs[idx]) if preds[idx] == 1 else float(1 - probs[idx]),
                        'probability': float(probs[idx])
                    })

            progress_bar.set_postfix({'misclassified': len(misclassified)})

    save_dir = os.path.join(OUTPUT_PATH, "misclassified_visualizations")
    os.makedirs(save_dir, exist_ok=True)
    for sample in misclassified[:5]:
        idx = test_loader.dataset.data.index[test_loader.dataset.data['image_path'] == sample['image_path']].tolist()[0]
        image, label, path = test_loader.dataset[idx]
        image_name = Path(path).name
        save_path = os.path.join(save_dir, f"{image_name}_true{sample['true_label']}_pred{sample['predicted_label']}.png")
        visualize_attention_map(model, image, label, sample['probability'], save_path, image_name)

    with open(os.path.join(OUTPUT_PATH, "misclassification_analysis.txt"), 'w') as f:
        f.write(f"MISCLASSIFICATION ANALYSIS - ImprovedXception\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total misclassified samples: {len(misclassified)}\n")

        fake_as_real = [m for m in misclassified if m['true_label'] == 'fake']
        real_as_fake = [m for m in misclassified if m['true_label'] == 'real']

        f.write(f"Fake misclassified as Real: {len(fake_as_real)}\n")
        f.write(f"Real misclassified as Fake: {len(real_as_fake)}\n\n")

        if fake_as_real:
            f.write("TOP 5 FAKE MISCLASSIFIED AS REAL:\n")
            for i, error in enumerate(sorted(fake_as_real, key=lambda x: x['confidence'], reverse=True)[:5], 1):
                f.write(f"{i}. {error['image_path']}\n")
                f.write(f"   Confidence: {error['confidence']:.4f}\n\n")

        if real_as_fake:
            f.write("TOP 5 REAL MISCLASSIFIED AS FAKE:\n")
            for i, error in enumerate(sorted(real_as_fake, key=lambda x: x['confidence'], reverse=True)[:5], 1):
                f.write(f"{i}. {error['image_path']}\n")
                f.write(f"   Confidence: {error['confidence']:.4f}\n\n")

    if misclassified:
        pd.DataFrame(misclassified).to_csv(os.path.join(OUTPUT_PATH, "misclassified_samples.csv"), index=False)

    return misclassified

def train_improved_xception():
    print("=" * 60)
    print("IMPROVED XCEPTION TRAINING FOR DEEPFAKE DETECTION")
    print("=" * 60)

    model = ImprovedXception(num_classes=1, dropout_rate=0.5).to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model created. Parameters: {params:.2f}M")

    print("\nLoading datasets...")
    train_dataset = DeepfakeDataset(os.path.join(DATA_PATH, "train.csv"), transform=train_transform)
    val_dataset = DeepfakeDataset(os.path.join(DATA_PATH, "val.csv"), transform=val_test_transform)
    test_dataset = DeepfakeDataset(os.path.join(DATA_PATH, "test.csv"), transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Dataset sizes - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")

    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    fake_ratio = sum(train_labels) / len(train_labels)
    print(f"Fake ratio in training set: {fake_ratio:.3f}")

    criterion = FocalLoss(alpha=ALPHA, gamma=GAMMA)
    scaler = GradScaler(enabled=DEVICE.type == 'cuda')
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [], 'train_auc': [], 'val_auc': [],
        'train_precision': [], 'val_precision': [], 'train_recall': [], 'val_recall': []
    }
    best_val_loss = float('inf')

    # Phase 1: Pretraining (freeze all except classifier)
    for name, param in model.named_parameters():
        if 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if not trainable_params:
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = list(model.parameters())

    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE)

    print("\nPhase 1: Pretraining...")
    for epoch in range(PRETRAIN_EPOCHS):
        print(f"  Epoch {epoch + 1}/{PRETRAIN_EPOCHS}")
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])

        print(
            f"    Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
            f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}, "
            f"Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}"
        )
        print(
            f"    Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}, "
            f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(MODEL_PATH, "best_model.pth"))

        if (epoch + 1) % SAVE_ATTENTION_FREQ == 0:
            print(f"  Saving attention visualizations for epoch {epoch + 1}...")
            save_attention_samples(model, train_loader, epoch + 1, "train", NUM_ATTENTION_SAMPLES)
            save_attention_samples(model, val_loader, epoch + 1, "val", NUM_ATTENTION_SAMPLES)

    # Phase 2: Finetuning
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE)

    print("\nPhase 2: Finetuning...")
    for epoch in range(FINETUNE_EPOCHS):
        print(f"  Epoch {epoch + 1}/{FINETUNE_EPOCHS}")
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])

        print(
            f"    Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
            f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}, "
            f"Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}"
        )
        print(
            f"    Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}, "
            f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(MODEL_PATH, "best_model.pth"))

        if (epoch + 1) % SAVE_ATTENTION_FREQ == 0:
            print(f"  Saving attention visualizations for epoch {epoch + 1}...")
            save_attention_samples(model, train_loader, epoch + PRETRAIN_EPOCHS + 1, "train", NUM_ATTENTION_SAMPLES)
            save_attention_samples(model, val_loader, epoch + PRETRAIN_EPOCHS + 1, "val", NUM_ATTENTION_SAMPLES)

    torch.save({'model_state_dict': model.state_dict()}, os.path.join(MODEL_PATH, "final_model.pth"))

    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(MODEL_PATH, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\nEvaluating on test set...")
    test_loss, test_metrics = validate_epoch(model, test_loader, criterion)

    model.eval()
    y_true, y_pred_prob = [], []
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc='Getting test predictions'):
            images = images.to(DEVICE)
            with autocast(enabled=DEVICE.type == 'cuda'):
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            y_true.extend(labels.numpy())
            y_pred_prob.extend(probs)

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    print("\nSaving attention visualizations for test set...")
    save_attention_samples(model, test_loader, PRETRAIN_EPOCHS + FINETUNE_EPOCHS, "test", NUM_ATTENTION_SAMPLES)

    print("\nMeasuring inference time...")
    start_time = time.time()
    total_samples = 0
    with torch.no_grad():
        for images, _, _ in tqdm(test_loader, desc='Inference timing'):
            images = images.to(DEVICE)
            with autocast(enabled=DEVICE.type == 'cuda'):
                _ = model(images)
            total_samples += images.size(0)
    total_time = time.time() - start_time
    inference_time_ms = (total_time / total_samples) * 1000
    fps = total_samples / total_time

    misclassified = analyze_misclassifications(model, test_loader)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"MCC: {test_metrics['mcc']:.4f}")
    print(f"Fake Detection Rate: {test_metrics['fake_detection_rate']:.4f}")
    print(f"Real Detection Rate: {test_metrics['real_detection_rate']:.4f}")
    print(f"Inference: {inference_time_ms:.2f} ms/sample ({fps:.1f} FPS)")
    print(f"Misclassified: {len(misclassified)} samples")
    if misclassified:
        print("Sample misclassifications:")
        for i, sample in enumerate(misclassified[:5], 1):
            print(
                f"  {i}. True: {sample['true_label']}, Pred: {sample['predicted_label']}, "
                f"Conf: {sample['confidence']:.3f}, Path: {sample['image_path']}")

    save_plots(history, test_metrics, y_true, y_pred_prob)

    results = {
        'model_name': 'ImprovedXception',
        'test_loss': test_loss,
        'accuracy': test_metrics['accuracy'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'f1': test_metrics['f1'],
        'auc': test_metrics['auc'],
        'mcc': test_metrics['mcc'],
        'fake_detection_rate': test_metrics['fake_detection_rate'],
        'real_detection_rate': test_metrics['real_detection_rate'],
        'inference_time_ms': inference_time_ms,
        'fps': fps,
        'parameters_millions': params,
        'total_misclassified': len(misclassified)
    }

    with open(os.path.join(OUTPUT_PATH, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(OUTPUT_PATH, "results.txt"), 'w') as f:
        f.write(f"Test Results - ImprovedXception\n")
        f.write(f"Loss: {test_loss:.4f}\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"F1: {test_metrics['f1']:.4f}\n")
        f.write(f"AUC: {test_metrics['auc']:.4f}\n")
        f.write(f"MCC: {test_metrics['mcc']:.4f}\n")
        f.write(f"Fake Detection Rate: {test_metrics['fake_detection_rate']:.4f}\n")
        f.write(f"Real Detection Rate: {test_metrics['real_detection_rate']:.4f}\n")
        f.write(f"Inference: {inference_time_ms:.2f} ms/sample ({fps:.1f} FPS)\n")
        f.write(f"Parameters: {params:.2f}M\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"Real: {test_metrics['true_negatives']} TN, {test_metrics['false_positives']} FP\n")
        f.write(f"Fake: {test_metrics['false_negatives']} FN, {test_metrics['true_positives']} TP\n")

    print(f"\n✓ All results saved to: {OUTPUT_PATH}")
    torch.cuda.empty_cache()
    return model, results

if __name__ == "__main__":
    model, results = train_improved_xception()
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)