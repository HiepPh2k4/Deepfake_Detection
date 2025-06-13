import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import librosa
import numpy as np
import cv2
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

# Constants
MODEL_PATH = "/classification/models/ver1/ver1/deepfake_audio_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
FRAME_DURATION = 0.1  # seconds
OUTPUT_DIR = "D:/Deepfake_Detection_project/data_preprocessing/temp_frames"

# Data transformations
test_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Load CRNN model
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = nn.functional.avg_pool2d(x, (x.size(2), 1))
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


def extract_audio_from_video(video_path):
    temp_audio = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
    os.system(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar {SAMPLE_RATE} -ac 1 {temp_audio}")
    return temp_audio


def audio_to_spectrogram(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Create spectrogram
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    # Convert to RGB image
    spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())
    spec_rgb = (spec_norm * 255).astype(np.uint8)
    spec_rgb = np.stack([spec_rgb] * 3, axis=2)

    # Split into frames
    frames = []
    frame_size = int(FRAME_DURATION * sr)
    for i in range(0, len(y) - frame_size, frame_size):
        frame_spec = spec_rgb[:, i:i + frame_size]
        if frame_spec.shape[1] == frame_size:
            frame_pil = Image.fromarray(frame_spec)
            frames.append(frame_pil)

    return frames


def predict_frames(frames, model):
    model.eval()
    predictions = []

    with torch.no_grad():
        for frame in frames:
            # Apply transformations
            frame = test_transforms(frame).unsqueeze(0).to(DEVICE)

            # Get prediction
            output = model(frame)
            prob = torch.softmax(output, dim=1)[0, 1].item()
            predictions.append(prob)

    # Average predictions
    avg_prob = np.mean(predictions) if predictions else 0.5
    return avg_prob


def main(input_path):
    # Create temp directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Check if input is video or audio
        is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov'))

        # Extract audio
        if is_video:
            audio_path = extract_audio_from_video(input_path)
        else:
            audio_path = input_path

        # Convert to spectrogram frames
        frames = audio_to_spectrogram(audio_path)

        if not frames:
            print("Không thể tạo frame từ audio!")
            return

        # Load model
        model = CRNN(num_classes=2).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

        # Predict
        prob = predict_frames(frames, model)

        # Display result
        print(f"\nKết quả dự đoán:")
        print(f"Xác suất là fake: {prob:.4f}")
        print(f"Kết luận: {'Fake' if prob > 0.5 else 'Real'}")

        # Clean upc
        if is_video:
            os.remove(audio_path)

    except Exception as e:
        print(f"Lỗi: {str(e)}")

    finally:
        # Clean up temp directory
        if os.path.exists(OUTPUT_DIR):
            for f in os.listdir(OUTPUT_DIR):
                os.remove(os.path.join(OUTPUT_DIR, f))
            os.rmdir(OUTPUT_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Deepfake Audio Detection")
    parser.add_argument(
        "D:/Deepfake_Detection_project/data/FakeAVCeleb_v1.2/RealVideo-FakeAudio/African/men/id00076/00109_fake.mp4",
        help="Path to video or audio file")
    args = parser.parse_args()
    main(args.input_path)