import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from transforms import DeepfakeDataset, train_transform, val_test_transform

# Constants
# DATA_PATH = "D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full"
# MODEL_PATH = "/classification/models_face/models_full"
# OUTPUT_PATH = "D:/Deepfake_Detection_project/classification/output/output_full"

DATA_PATH = "G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_test_remote"
MODEL_PATH = "G:/Hiep/Deepfake_Detection/classification/models/model_remote_testt"
OUTPUT_PATH = "G:/Hiep/Deepfake_Detection/classification/output/output_remote_testt"

BATCH_SIZE = 32
NUM_WORKERS = 6
PRETRAIN_EPOCHS = 3
FINETUNE_EPOCHS = 15
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 0.25
PATIENCE = 7

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model():
    """Create and modify Xception model for binary classification."""
    model = timm.create_model("xception", pretrained=True, num_classes=1)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
    )
    return model.to(DEVICE)


def compute_metrics(y_true, y_pred, y_prob):
    """Compute evaluation metrics for binary classification."""
    y_pred = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "cm": cm,
        "accuracy": (y_pred == y_true).mean(),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "ap": average_precision_score(y_true, y_prob),
    }


def train_epoch(model, loader, criterion, optimizer):
    """Run one training epoch."""
    model.train()
    total_loss, labels, probs = 0, [], []
    for images, targets in tqdm(loader, desc="Training"):
        images, targets = images.to(DEVICE), targets.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
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


def train_model(model, train_loader, val_loader, epochs, save_path):
    """Train model with early stopping and learning rate scheduling."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5) #0.5 vaf 3 tot hon
    best_val_accuracy = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        # Train
        train_loss, train_labels, train_probs = train_epoch(model, train_loader, criterion, optimizer)
        train_metrics = compute_metrics(train_labels, train_probs, train_probs)

        # Validate
        val_loss, val_labels, val_probs = validate_epoch(model, val_loader, criterion)
        val_metrics = compute_metrics(val_labels, val_probs, val_probs)

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        # Print progress
        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Loss={train_loss:.4f}, Acc={train_metrics['accuracy']:.4f}, "
            f"Val Loss={val_loss:.4f}, Acc={val_metrics['accuracy']:.4f}"
        )

        # Save best model
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return history


def evaluate_model(model, test_loader, criterion):
    """Evaluate model on test set."""
    test_loss, test_labels, test_probs = validate_epoch(model, test_loader, criterion)
    metrics = compute_metrics(test_labels, test_probs, test_probs)
    return test_loss, metrics, test_labels, test_probs


def plot_results(history, metrics, test_labels, test_probs, output_path):
    """Plot training history, confusion matrix, and precision-recall curve."""
    # Training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{output_path}/training_plots.png")
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{output_path}/confusion_matrix.png")
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(test_labels, test_probs)
    plt.figure()
    plt.plot(recall, precision, label=f"PR Curve (AP={metrics['ap']:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(f"{output_path}/pr_curve.png")
    plt.close()


def main():
    """Main function to run training and evaluation."""
    # Load datasets
    train_dataset = DeepfakeDataset(f"{DATA_PATH}/train.csv", transform=train_transform)
    val_dataset = DeepfakeDataset(f"{DATA_PATH}/val.csv", transform=val_test_transform)
    test_dataset = DeepfakeDataset(f"{DATA_PATH}/test.csv", transform=val_test_transform)
    print(
        f"Train: {len(train_dataset)} images, "
        f"Validation: {len(val_dataset)} images, "
        f"Test: {len(test_dataset)} images"
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # Initialize model
    model = create_model()

    # Pre-training (freeze backbone)
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    history_pretrain = train_model(
        model, train_loader, val_loader, PRETRAIN_EPOCHS, f"{MODEL_PATH}/deepfake_model_pretrain.pt"
    )

    # Fine-tuning (unfreeze all)
    for param in model.parameters():
        param.requires_grad = True
    history_finetune = train_model(
        model, train_loader, val_loader, FINETUNE_EPOCHS, f"{MODEL_PATH}/deepfake_model_best.pt"
    )

    # Save final model
    torch.save(model.state_dict(), f"{MODEL_PATH}/deepfake_model_final.pt")

    # Combine history
    history = {
        k: history_pretrain[k] + history_finetune[k] for k in history_pretrain
    }

    # Evaluate on test set
    best_model = create_model()
    best_model.load_state_dict(torch.load(f"{MODEL_PATH}/deepfake_model_final.pt"))
    criterion = nn.BCEWithLogitsLoss()
    test_loss, metrics, test_labels, test_probs = evaluate_model(best_model, test_loader, criterion)

    # Print test results
    print("\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"Precision: {metrics['precision']*100:.1f}%")
    print(f"Recall: {metrics['recall']*100:.1f}%")
    print(f"F1 Score: {metrics['f1']*100:.1f}%")
    print(f"AUC: {metrics['auc']*100:.1f}%")
    print(f"Average Precision: {metrics['ap']*100:.1f}%")
    print(f"Confusion Matrix:\n{metrics['cm']}")

    # Plot results
    plot_results(history, metrics, test_labels, test_probs, OUTPUT_PATH)


if __name__ == "__main__":
    main()

