import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# Define transforms
IMG_SIZE = 299
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Define DeepfakeDataset
class DeepfakeDataset(Dataset):
    """Custom Dataset for deepfake detection."""
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        label = 0 if self.data.iloc[idx]['label'] == 'real' else 1
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, float(label), img_path

# Custom collate function
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float)
    img_paths = [item[2] for item in batch]
    return images, labels, img_paths

# Constants
DATA_PATH = "G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_test_remote"
MODEL_PATH = "G:/Hiep/Deepfake_Detection/classification/models/model_remote_test"
OUTPUT_PATH = "G:/Hiep/Deepfake_Detection/classification/output/output_remote_test"

BATCH_SIZE = 32
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model():
    """Create and modify Xception model for binary classification."""
    model = timm.create_model("xception", pretrained=False, num_classes=1)
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

def evaluate_model(model, test_loader, criterion):
    """Evaluate model on test set and collect misclassified samples."""
    model.eval()
    total_loss, labels, probs, paths, misclassified = 0, [], [], [], []
    with torch.no_grad():
        for images, targets, img_paths in tqdm(test_loader, desc="Evaluating"):
            images, targets = images.to(DEVICE), targets.to(DEVICE).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            probs_batch = torch.sigmoid(outputs).cpu().numpy().flatten()
            labels_batch = targets.cpu().numpy().flatten()
            preds_batch = (probs_batch > 0.5).astype(int)

            # Collect misclassified samples
            for i in range(len(labels_batch)):
                if preds_batch[i] != labels_batch[i]:
                    misclassified.append({
                        'path': img_paths[i],
                        'true_label': labels_batch[i],
                        'predicted_label': preds_batch[i],
                        'probability': probs_batch[i]
                    })

            probs.extend(probs_batch)
            labels.extend(labels_batch)
            paths.extend(img_paths)

    test_loss = total_loss / len(test_loader.dataset)
    metrics = compute_metrics(np.array(labels), np.array(probs), np.array(probs))
    return test_loss, metrics, np.array(labels), np.array(probs), misclassified

def plot_results(metrics, test_labels, test_probs, output_path):
    """Plot confusion matrix and precision-recall curve."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{output_path}/confusion_matrix.png")
    plt.close()

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
    """Main function to evaluate the model on the test set."""
    test_dataset = DeepfakeDataset(f"{DATA_PATH}/test.csv", transform=val_test_transform)
    print(f"Test: {len(test_dataset)} images")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    model = create_model()
    model.load_state_dict(torch.load(f"{MODEL_PATH}/deepfake_model_final.pt"))
    criterion = nn.BCEWithLogitsLoss()

    test_loss, metrics, test_labels, test_probs, misclassified = evaluate_model(model, test_loader, criterion)

    print("\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"Precision: {metrics['precision']*100:.1f}%")
    print(f"Recall: {metrics['recall']*100:.1f}%")
    print(f"F1 Score: {metrics['f1']*100:.1f}%")
    print(f"AUC: {metrics['auc']*100:.1f}%")
    print(f"Average Precision: {metrics['ap']*100:.1f}%")
    print(f"Confusion Matrix:\n{metrics['cm']}")

    with open(f"{OUTPUT_PATH}/misclassified_samples.txt", "w") as f:
        for sample in misclassified:
            f.write(f"Image Path: {sample['path']}\n")
            f.write(f"True Label: {sample['true_label']}\n")
            f.write(f"Predicted Label: {sample['predicted_label']}\n")
            f.write(f"Probability: {sample['probability']:.4f}\n")
            f.write("-" * 50 + "\n")
    print(f"\nMisclassified samples saved to {OUTPUT_PATH}/misclassified_samples.txt")

    plot_results(metrics, test_labels, test_probs, OUTPUT_PATH)

if __name__ == "__main__":
    main()