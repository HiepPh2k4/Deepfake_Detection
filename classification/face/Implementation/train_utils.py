import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from tqdm import tqdm
import time

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        if self.alpha is not None:
            alpha_t = self.alpha[0] * (1 - targets) + self.alpha[1] * targets  # alpha_0 for REAL (0), alpha_1 for FAKE (1)
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, model_name, num_epochs, device, save_dir, patience=5):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)

    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]', leave=False):
            images, labels = images.to(device), labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]', leave=False):
                images, labels = images.to(device), labels.to(device).view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Save final model
    os.makedirs(os.path.dirname(os.path.join(save_dir, f'{model_name}_deepfake.pth')), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_deepfake.pth'))
    print(f"Saved final model {model_name}!")

    # Evaluation on test set using final model
    model.eval()
    y_true, y_pred, y_pred_prob, img_paths = [], [], [], []
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images, labels = images.to(device), labels.to(device).view(-1, 1)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(predicted.cpu().numpy().flatten())
            y_pred_prob.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            img_paths.extend(paths)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for images, _, _ in test_loader:
            images = images.to(device)
            _ = model(images)
    inference_time = (time.time() - start_time) / len(test_loader.dataset) * 1000  # ms/sample
    print(f"Inference time: {inference_time:.2f} ms/sample")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_prob = np.array(y_pred_prob)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
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
        'true_positives': tp,
        'inference_time_ms': inference_time
    }

    print(f"\nEvaluation results for {model_name} on test set:")
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, 'plots', f'{model_name}_confusion_matrix.png'))
    plt.close()

    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'plots', f'{model_name}_roc_curve.png'))
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - {model_name}')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Curve - {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plots', f'{model_name}_loss_accuracy.png'))
    plt.close()

    misclassified = [(img_paths[i], y_true[i], y_pred[i], y_pred_prob[i]) for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    print(f"\nNumber of misclassified samples: {len(misclassified)}")

    misclassified_txt_path = os.path.join(save_dir, 'plots', f'{model_name}_misclassifications.txt')
    with open(misclassified_txt_path, 'w', encoding='utf-8') as f:
        f.write("List of misclassified samples:\n")
        f.write("Image Path | True Label | Predicted Label | Predicted Probability\n")
        f.write("-" * 80 + "\n")
        for img_path, true_label, pred_label, pred_prob in misclassified:
            true_label_str = "Real" if true_label == 0 else "Fake"
            pred_label_str = "Real" if pred_label == 0 else "Fake"
            f.write(f"{img_path} | {true_label_str} | {pred_label_str} | {pred_prob:.4f}\n")

    if len(misclassified) > 0:
        print("Some misclassified samples (displaying up to 5):")
        plt.figure(figsize=(15, 5))
        for i, (img_path, true_label, pred_label, pred_prob) in enumerate(misclassified[:5]):
            img = Image.open(img_path).convert('RGB')
            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.title(f'True: {"Real" if true_label == 0 else "Fake"}\nPred: {"Real" if pred_label == 0 else "Fake"} ({pred_prob:.2f})')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'plots', f'{model_name}_misclassifications.png'))
        plt.close()

    return model, metrics, train_losses, val_losses, train_accs, val_accs