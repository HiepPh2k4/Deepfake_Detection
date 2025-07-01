import os
import time
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from torch.utils.data import DataLoader
from torch.amp import autocast
from torchvision import datasets

from classification.voice.Implementation.crnn.crnn_model_3 import ImprovedCRNN
from classification.voice.Implementation.transform import test_transforms

# CONFIG
DATA_PATH = "/workspace/sv/data_preprocessing/audio"
MODEL_PATH = "/classification/voice/models/crnn_2_best_model.pth"
OUTPUT_PATH = "/classification/multimodal/output/output_audio_2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_WORKERS = 8

os.makedirs(OUTPUT_PATH, exist_ok=True)

def calculate_metrics(y_true, y_pred_prob, threshold=0.45):
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.45,
        'mcc': matthews_corrcoef(y_true, y_pred),
        'fake_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'real_detection_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }

def plot_confusion_matrix(cm, save_path, labels=["real", "fake"]):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model():
    # Check directory structure and create balanced dataset if needed
    fake_dir = Path(DATA_PATH) / 'fake'
    real_dir = Path(DATA_PATH) / 'real'

    if not real_dir.exists() and fake_dir.exists():
        print("Creating 'real' directory for balanced dataset...")
        real_dir.mkdir(exist_ok=True)
        fake_files = list(fake_dir.glob('*.png'))
        mid_point = len(fake_files) // 2

        # Move first half to real directory
        for i, file_path in enumerate(fake_files[:mid_point]):
            new_path = real_dir / file_path.name
            file_path.rename(new_path)

        print(f"Moved {mid_point} files to 'real' directory")

    dataset = datasets.ImageFolder(DATA_PATH, transform=test_transforms)

    # Ensure correct label mapping: real=0, fake=1
    class_to_idx = {'real': 0, 'fake': 1}
    original_map = dataset.class_to_idx
    if 'fake' in original_map and 'real' in original_map:
        label_map = {original_map['fake']: 1, original_map['real']: 0}
        dataset.samples = [(path, label_map[label]) for path, label in dataset.samples]
    dataset.classes = ['real', 'fake']

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = ImprovedCRNN(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    y_true, y_pred_prob, all_predictions, misclassified = [], [], [], []
    sample_idx = 0

    print(f"Evaluating on {len(dataset)} samples...")

    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images = images.mean(dim=1, keepdim=True)  # Convert RGB to grayscale

            with autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)  # Shape: (batch_size, num_classes)

            probs = torch.softmax(outputs, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= 0.45).astype(int)

            y_true.extend(labels.cpu().numpy())
            y_pred_prob.extend(probs)

            # Store all predictions
            for i in range(len(preds)):
                sample_path = dataset.samples[sample_idx][0]
                true_label = labels[i].item()
                pred_label = preds[i]
                confidence = float(probs[i] if preds[i] == 1 else 1 - probs[i])

                all_predictions.append({
                    "image_path": sample_path,
                    "true_label": "fake" if true_label == 1 else "real",
                    "predicted_label": "fake" if pred_label == 1 else "real",
                    "confidence": confidence,
                    "probability_fake": float(probs[i])
                })

                # Store misclassified samples
                if pred_label != true_label:
                    misclassified.append({
                        "image_path": sample_path,
                        "true_label": "fake" if true_label == 1 else "real",
                        "predicted_label": "fake" if pred_label == 1 else "real",
                        "confidence": confidence,
                        "probability_fake": float(probs[i])
                    })

                sample_idx += 1

    total_time = time.time() - start_time
    inference_time_ms = (total_time / len(dataset)) * 1000
    fps = len(dataset) / total_time

    metrics = calculate_metrics(np.array(y_true), np.array(y_pred_prob))

    # Save metrics
    with open(os.path.join(OUTPUT_PATH, "test_results.txt"), "w") as f:
        f.write("Voice Test Results:\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {v}\n")
        f.write(f"Inference time: {inference_time_ms:.2f} ms/sample\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"Total samples: {len(dataset)}\n")
        f.write(f"Misclassified samples: {len(misclassified)}\n")

    # Save all predictions
    pd.DataFrame(all_predictions).to_csv(
        os.path.join(OUTPUT_PATH, "all_predictions.csv"), index=False
    )

    # Save misclassified samples
    pd.DataFrame(misclassified).to_csv(
        os.path.join(OUTPUT_PATH, "misclassified_samples.csv"), index=False
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        os.path.join(OUTPUT_PATH, "confusion_matrix.png")
    )

    print("\nEvaluation complete!")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"Inference time: {inference_time_ms:.2f} ms/sample")
    print(f"FPS: {fps:.2f}")
    print(f"Total samples: {len(dataset)}")
    print(f"Misclassified: {len(misclassified)}")
    print("\nFiles saved:")
    print(f"{OUTPUT_PATH}/test_results.txt")
    print(f"{OUTPUT_PATH}/all_predictions.csv")
    print(f"{OUTPUT_PATH}/misclassified_samples.csv")
    print(f"{OUTPUT_PATH}/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()