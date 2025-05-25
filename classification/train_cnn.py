import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from transforms import DeepfakeDataset, transform
from model_custom import CustomCNN
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Metrics computation
def compute_metrics(y_true, y_pred, y_prob):
    y_pred = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    accuracy = (y_pred == y_true).mean()
    return cm, f1, precision, recall, auc, accuracy

# Training function with mixed precision
def train_model(model, train_loader, val_loader, epochs, save_path, patience=10):
    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.0]).to(device))
    scaler = torch.amp.GradScaler('cuda')
    best_val_accuracy = -float('inf')
    patience_counter = 0
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [],
               'precision': [], 'val_precision': [], 'recall': [], 'val_recall': [],
               'auc': [], 'val_auc': []}

    for epoch in range(epochs):
        model.train()
        train_loss, train_preds, train_probs, train_labels = 0.0, [], [], []
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Train"):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())
            train_preds.extend((torch.sigmoid(outputs) > 0.5).float().detach().cpu().numpy().flatten())
            train_labels.extend(labels.cpu().numpy().flatten())
            torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        _, _, train_precision, train_recall, train_auc, train_accuracy = compute_metrics(
            np.array(train_labels), np.array(train_preds), np.array(train_probs))

        model.eval()
        val_loss, val_preds, val_probs, val_labels = 0.0, [], [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Val"):
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                val_preds.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())
                torch.cuda.empty_cache()

        val_loss /= len(val_loader)
        _, _, val_precision, val_recall, val_auc, val_accuracy = compute_metrics(
            np.array(val_labels), np.array(val_preds), np.array(val_probs))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        history['auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_accuracy:.4f}, Prec={train_precision:.4f}, "
              f"Rec={train_recall:.4f}, AUC={train_auc:.4f}")
        print(f"  Val: Loss={val_loss:.4f}, Acc={val_accuracy:.4f}, Prec={val_precision:.4f}, "
              f"Rec={val_recall:.4f}, AUC={val_auc:.4f}")
        print(f"  Current best val accuracy: {best_val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model at {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return history

# Main execution block
if __name__ == '__main__':
    # Load data
    base_path = "D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full"
    train_dataset = DeepfakeDataset(f"{base_path}/train_rgb.csv", transform=transform)
    val_dataset = DeepfakeDataset(f"{base_path}/val_rgb.csv", transform=transform)
    test_dataset = DeepfakeDataset(f"{base_path}/test_rgb.csv", transform=transform)

    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    # DataLoaders with optimization
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model paths
    model_base_path = "D:/Deepfake_Detection_project/classification/models/models_full"
    output_base_path = "D:/Deepfake_Detection_project/classification/output/output_full"

    # Create directories if they don't exist
    os.makedirs(model_base_path, exist_ok=True)
    os.makedirs(output_base_path, exist_ok=True)

    # Initialize model
    model = CustomCNN().to(device)

    # Pre-training (3 epochs, freeze layers except exit flow)
    for name, param in model.named_parameters():
        if 'exit_flow' not in name:
            param.requires_grad = False

    pretrain_save_path = f"{model_base_path}/deepfake_model_pretrain.pt"
    history_pretrain = train_model(
        model, train_loader, val_loader, epochs=3,
        save_path=pretrain_save_path
    )

    # Always save the model after pre-training
    torch.save(model.state_dict(), pretrain_save_path)
    print(f"Saved pre-trained model at {pretrain_save_path}")

    # Fine-tuning (17 epochs, unfreeze all layers)
    for param in model.parameters():
        param.requires_grad = True

    finetune_save_path = f"{model_base_path}/deepfake_model_best.pt"
    history_finetune = train_model(
        model, train_loader, val_loader, epochs=17,
        save_path=finetune_save_path, patience=10
    )

    # Save final model
    final_save_path = f"{model_base_path}/deepfake_model_final.pt"
    torch.save(model.state_dict(), final_save_path)
    print(f"Saved final model at {final_save_path}")

    # Load best model for evaluation
    best_model = CustomCNN().to(device)
    best_model.load_state_dict(torch.load(finetune_save_path))
    best_model.eval()

    # Evaluate on test set
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_preds, test_probs, test_labels = 0.0, [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating test set"):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            with autocast():
                outputs = best_model(images)
                loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            test_preds.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy().flatten())
            test_labels.extend(labels.cpu().numpy().flatten())
            torch.cuda.empty_cache()

    test_loss /= len(test_loader)
    cm, f1, precision, recall, auc, accuracy = compute_metrics(
        np.array(test_labels), np.array(test_preds), np.array(test_probs))

    # Print test metrics
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Precision: {precision * 100:.2f}%")
    print(f"Test Recall: {recall * 100:.2f}%")
    print(f"Test AUC: {auc * 100:.2f}%")
    print(f"Test F1-score: {f1 * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{output_base_path}/confusion_matrix.png")
    plt.show()

    # Combine and plot training history
    history_combined = {key: history_pretrain[key] + history_finetune[key] for key in history_pretrain}
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_combined['loss'], label='Train Loss')
    plt.plot(history_combined['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_combined['accuracy'], label='Train Accuracy')
    plt.plot(history_combined['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(f"{output_base_path}/training_plots.png")
    plt.show()