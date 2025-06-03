import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import os

from transforms_advanced import DeepfakeDataset, train_transform, val_test_transform
from model_cnn import HybridDeepfakeCNN, FocalLoss

# Constants - OPTIMIZED FOR >92% ACCURACY
DATA_PATH = "G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_test_remote"
MODEL_PATH = "G:/Hiep/Deepfake_Detection/classification/models/model_remote_test_cnn"
OUTPUT_PATH = "G:/Hiep/Deepfake_Detection/classification/output/output_remote_test_cnn"

BATCH_SIZE = 32  # Back to stable batch size
NUM_WORKERS = 6
PRETRAIN_EPOCHS = 3  # Same as original for comparison
FINETUNE_EPOCHS = 15  # Same as original for comparison
LEARNING_RATE = 2e-4  # Back to effective learning rate
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 0.17  # Correct class weight for imbalanced data
PATIENCE = 8

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_balanced_sampler(dataset):
    """Create balanced sampler to handle class imbalance"""
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)

    labels = np.array(labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    weights = class_weights[labels]

    return WeightedRandomSampler(weights, len(weights))


def create_model():
    """Create optimized Hybrid CNN model for >92% accuracy."""
    model = HybridDeepfakeCNN(num_classes=1, dropout_rate=0.4)  # Higher dropout
    return model.to(DEVICE)


def compute_metrics(y_true, y_pred, y_prob):
    """Compute evaluation metrics for binary classification."""
    y_pred = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "cm": cm,
        "accuracy": (y_pred == y_true).mean(),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0,
        "ap": average_precision_score(y_true, y_prob),
    }


def train_epoch(model, loader, criterion, optimizer):
    """Run one training epoch with improved stability."""
    model.train()
    total_loss, labels, probs = 0, [], []

    # Fixed AMP usage for newer PyTorch versions
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    for images, targets in tqdm(loader, desc="Training"):
        images, targets = images.to(DEVICE), targets.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        probs.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())
        labels.extend(targets.cpu().numpy().flatten())

    return total_loss / len(loader.dataset), np.array(labels), np.array(probs)


def validate_epoch(model, loader, criterion):
    """Run one validation epoch."""
    model.eval()
    total_loss, labels, probs = 0, [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            labels.extend(targets.cpu().numpy().flatten())
    return total_loss / len(loader.dataset), np.array(labels), np.array(probs)


def train_model(model, train_loader, val_loader, epochs, save_path, phase_name=""):
    """Train model with advanced techniques for >92% accuracy."""
    # Optimized optimizer and scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8
    )

    # Standard BCE loss with pos_weight (more stable than Focal)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE))

    # No scheduler - fixed learning rate for stability
    # scheduler = None

    best_val_accuracy = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n🚀 Starting {phase_name} - Target: >92% Accuracy")
    print("-" * 60)

    for epoch in range(epochs):
        # Train
        train_loss, train_labels, train_probs = train_epoch(model, train_loader, criterion, optimizer)
        train_metrics = compute_metrics(train_labels, train_probs, train_probs)

        # Validate
        val_loss, val_labels, val_probs = validate_epoch(model, val_loader, criterion)
        val_metrics = compute_metrics(val_labels, val_probs, val_probs)

        # No scheduler update - using fixed learning rate

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        # Fixed learning rate
        current_lr = LEARNING_RATE
        print(
            f"Epoch {epoch + 1:2d}/{epochs}: "
            f"Train Loss={train_loss:.4f}, Acc={train_metrics['accuracy']:.1%}, "
            f"Val Loss={val_loss:.4f}, Acc={val_metrics['accuracy']:.1%}, "
            f"AUC={val_metrics['auc']:.3f}, LR={current_lr:.2e}"
        )

        # Target achievement indicators
        if val_metrics['accuracy'] >= 0.92:
            print(f"    🎯 TARGET ACHIEVED! Accuracy: {val_metrics['accuracy']:.1%}")
        if val_metrics['auc'] >= 0.90:
            print(f"    🚀 EXCELLENT AUC: {val_metrics['auc']:.3f}")

        # Save best model based on accuracy
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(model.state_dict(), save_path)
            print(f"    💾 Best model saved! Accuracy: {best_val_accuracy:.1%}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping at epoch {epoch + 1}")
            break

    print(f"✅ {phase_name} completed! Best accuracy: {best_val_accuracy:.1%}")
    return history


def evaluate_model(model, test_loader, criterion):
    """Evaluate model on test set."""
    test_loss, test_labels, test_probs = validate_epoch(model, test_loader, criterion)
    metrics = compute_metrics(test_labels, test_probs, test_probs)
    return test_loss, metrics, test_labels, test_probs


def plot_results(history, metrics, test_labels, test_probs, output_path):
    """Plot training history, confusion matrix, and precision-recall curve."""
    # Training history
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Val Loss", linewidth=2)
    plt.title("Training Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(history["train_acc"], label="Train Acc", linewidth=2)
    plt.plot(history["val_acc"], label="Val Acc", linewidth=2)
    plt.axhline(y=0.92, color='red', linestyle='--', alpha=0.7, label='92% Target')
    plt.title("Accuracy Progress", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Final accuracy bar
    plt.subplot(1, 3, 3)
    final_acc = max(history["val_acc"])
    color = 'green' if final_acc >= 0.92 else 'orange' if final_acc >= 0.85 else 'red'
    plt.bar(['Final Accuracy'], [final_acc], color=color, alpha=0.7)
    plt.axhline(y=0.92, color='red', linestyle='--', alpha=0.7, label='92% Target')
    plt.title("Final Performance", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()

    # Add percentage text on bar
    plt.text(0, final_acc + 0.02, f'{final_acc:.1%}',
             ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_path}/training_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Enhanced Confusion matrix
    plt.figure(figsize=(8, 6))
    cm_percent = metrics["cm"] / metrics["cm"].sum() * 100

    # Create custom annotations
    annotations = []
    for i in range(2):
        row = []
        for j in range(2):
            count = metrics["cm"][i, j]
            percent = cm_percent[i, j]
            row.append(f'{count}\n({percent:.1f}%)')
        annotations.append(row)

    sns.heatmap(metrics["cm"], annot=annotations, fmt='', cmap="Blues",
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'})
    plt.title("Confusion Matrix (Count and Percentage)", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"{output_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(test_labels, test_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=3, label=f'PR Curve (AP={metrics["ap"]:.3f})')
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_path}/pr_curve.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to run training and evaluation."""
    # Create output directories
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("🔥 HYBRID DEEPFAKE CNN - OPTIMIZED FOR >92% ACCURACY")
    print(f"🖥️  Device: {DEVICE}")
    print("=" * 60)

    # Load datasets
    train_dataset = DeepfakeDataset(f"{DATA_PATH}/train.csv", transform=train_transform)
    val_dataset = DeepfakeDataset(f"{DATA_PATH}/val.csv", transform=val_test_transform)
    test_dataset = DeepfakeDataset(f"{DATA_PATH}/test.csv", transform=val_test_transform)

    print(f"📁 Dataset loaded:")
    print(f"Train: {len(train_dataset):,} images")
    print(f"Validation: {len(val_dataset):,} images")
    print(f"Test: {len(test_dataset):,} images")

    # Create balanced data loaders
    train_sampler = create_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Initialize model
    model = create_model()

    # Phase 1: Pre-training (freeze backbone)
    print(f"\n🔒 PHASE 1: PRE-TRAINING (Backbone Frozen)")
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    history_pretrain = train_model(
        model, train_loader, val_loader, PRETRAIN_EPOCHS,
        f"{MODEL_PATH}/deepfake_model_pretrain.pt", "Pre-training"
    )

    # Phase 2: Fine-tuning (unfreeze all)
    print(f"\n🔓 PHASE 2: FINE-TUNING (Full Model)")
    for param in model.parameters():
        param.requires_grad = True

    history_finetune = train_model(
        model, train_loader, val_loader, FINETUNE_EPOCHS,
        f"{MODEL_PATH}/deepfake_model_best.pt", "Fine-tuning"
    )

    # Save final model
    torch.save(model.state_dict(), f"{MODEL_PATH}/deepfake_model_final.pt")

    # Combine training history
    history = {
        k: history_pretrain[k] + history_finetune[k] for k in history_pretrain
    }

    # Final evaluation
    print(f"\n🏆 FINAL EVALUATION")
    print("=" * 60)

    best_model = create_model()
    best_model.load_state_dict(torch.load(f"{MODEL_PATH}/deepfake_model_best.pt"))
    criterion = nn.BCEWithLogitsLoss()
    test_loss, metrics, test_labels, test_probs = evaluate_model(best_model, test_loader, criterion)

    # Print comprehensive test results
    print(f"\n📊 TEST RESULTS:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.1f}%")
    print(f"Precision: {metrics['precision'] * 100:.1f}%")
    print(f"Recall: {metrics['recall'] * 100:.1f}%")
    print(f"F1 Score: {metrics['f1'] * 100:.1f}%")
    print(f"AUC: {metrics['auc'] * 100:.1f}%")
    print(f"Average Precision: {metrics['ap'] * 100:.1f}%")

    print(f"\n📈 Confusion Matrix:")
    cm = metrics['cm']
    print(f"              Predicted")
    print(f"              Real  Fake")
    print(f"Actual Real:  {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f"Actual Fake:  {cm[1, 0]:4d}  {cm[1, 1]:4d}")

    # Success indicators
    if metrics['accuracy'] >= 0.92:
        print(f"\n🎉 SUCCESS! Model achieved >92% accuracy target!")
        print(f"   Final accuracy: {metrics['accuracy'] * 100:.1f}%")
    else:
        print(f"\n⚠️  Model achieved {metrics['accuracy'] * 100:.1f}% accuracy")
        print(f"   Target: >92% accuracy")

    if metrics['auc'] >= 0.90:
        print(f"🌟 Excellent AUC score: {metrics['auc'] * 100:.1f}%")

    # Generate plots
    plot_results(history, metrics, test_labels, test_probs, OUTPUT_PATH)

    print(f"\n✅ Training completed!")
    print(f"📊 Results saved to: {OUTPUT_PATH}")
    print(f"💾 Models saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()