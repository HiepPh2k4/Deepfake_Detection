import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data_generator import data_generator
from train_utils import train_and_evaluate, FocalLoss
import os

if __name__ == '__main__':
    # Set parameters
    DATA_PATH = "G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_full_remote"
    MODEL_PATH = "G:/Hiep/Deepfake_Detection/classification/models/model_shufflenet_v2"
    OUTPUT_PATH = "G:/Hiep/Deepfake_Detection/classification/output/output_shufflenet_v2"
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4
    ALPHA = 0.80  # Weight for REAL (0), so 1 - ALPHA = 0.20 for FAKE (1)
    GAMMA = 2.0
    PATIENCE = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_EPOCHS = 18

    # Load data
    train_csv = os.path.join(DATA_PATH, "train_rgb.csv")
    val_csv = os.path.join(DATA_PATH, "val_rgb.csv")
    test_csv = os.path.join(DATA_PATH, "test_rgb.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Number of training images: {len(train_df)}")
    print(f"Number of validation images: {len(val_df)}")
    print(f"Number of test images: {len(test_df)}")

    # Prepare ShuffleNet V2 model
    model = models.shufflenet_v2_x1_0(weights='ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(DEVICE)

    # Compile model
    criterion = FocalLoss(alpha=torch.tensor([ALPHA, 1 - ALPHA]), gamma=GAMMA)  # Prioritize REAL (0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Train and evaluate
    train_loader = data_generator(train_csv, BATCH_SIZE)
    val_loader = data_generator(val_csv, BATCH_SIZE)
    test_loader = data_generator(test_csv, BATCH_SIZE)

    model, metrics, _, _, _, _ = train_and_evaluate(
        model, train_loader, val_loader, test_loader, criterion, optimizer,
        model_name="ShuffleNetV2", num_epochs=NUM_EPOCHS, device=DEVICE, save_dir=OUTPUT_PATH, patience=PATIENCE
    )

    # Save final model
    os.makedirs(MODEL_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, "shufflenet_v2_deepfake.pth"))
    print("Saved model ShuffleNetV2!")