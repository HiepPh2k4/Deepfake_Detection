import cv2
import numpy as np
import torch
import timm
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from PIL import Image
import sys
from advanced_transform import val_test_transform

# Constants
MODEL_PATH = "D:/Deepfake_Detection_project/classification/models_face/models_full/deepfake_model_best.pt"
IMG_SIZE = 299
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepfakeGUI(QMainWindow):
    """GUI for deepfake detection with video and real-time webcam support."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deepfake Detection")
        self.setGeometry(100, 100, 800, 600)

        # Initialize model
        self.model = self.load_model()
        self.model.eval()

        # Webcam
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # GUI components
        self.init_ui()

    def load_model(self):
        """Load pre-trained Xception model."""
        model = timm.create_model("xception", pretrained=False, num_classes=1)
        # Do not modify fc, since the saved weights expect a single Linear layer
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        return model.to(DEVICE)

    def init_ui(self):
        """Initialize GUI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Video display
        self.video_label = QLabel("Video/Frame will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid black;")
        layout.addWidget(self.video_label)

        # Result display
        self.result_label = QLabel("Prediction: None")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        # Buttons
        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_video)
        layout.addWidget(self.select_button)

        self.webcam_button = QPushButton("Start Webcam")
        self.webcam_button.clicked.connect(self.toggle_webcam)
        layout.addWidget(self.webcam_button)

        # Drag and drop
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        """Handle drag enter event for video files."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle drop event for video files."""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith((".mp4", ".avi", ".mov")):
                self.process_video(file_path)
                break

    def select_video(self):
        """Open file dialog to select a video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_path:
            self.process_video(file_path)

    def process_video(self, video_path):
        """Process video and predict real/fake."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.result_label.setText("Error: Cannot open video")
            return

        probs = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # Process every 10th frame to speed up
            if frame_count % 10 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prob = self.predict_frame(frame_rgb)
                probs.append(prob)

                # Display frame
                self.display_frame(frame_rgb)

        cap.release()
        if probs:
            avg_prob = np.mean(probs)
            label = "Fake" if avg_prob > 0.5 else "Real"
            self.result_label.setText(f"Prediction: {label} ({avg_prob*100:.1f}%)")
        else:
            self.result_label.setText("Error: No valid frames")

    def predict_frame(self, frame_rgb):
        """Predict real/fake for a single frame."""
        image = Image.fromarray(frame_rgb)
        image = val_test_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = self.model(image)
            prob = torch.sigmoid(output).item()
        return prob

    def display_frame(self, frame_rgb):
        """Display frame in GUI."""
        frame_rgb = cv2.resize(frame_rgb, (640, 480))
        height, width, _ = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def toggle_webcam(self):
        """Start or stop webcam for real-time detection."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.result_label.setText("Error: Cannot open webcam")
                self.cap = None
                return
            self.webcam_button.setText("Stop Webcam")
            self.timer.start(100)  # Update every 100ms
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.webcam_button.setText("Start Webcam")
            self.video_label.setPixmap(QPixmap())
            self.result_label.setText("Prediction: None")

    def update_frame(self):
        """Update webcam frame and predict in real-time."""
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prob = self.predict_frame(frame_rgb)
            label = "Fake" if prob > 0.5 else "Real"
            self.result_label.setText(f"Prediction: {label} ({prob*100:.1f}%)")
            self.display_frame(frame_rgb)

    def closeEvent(self, event):
        """Clean up when closing window."""
        if self.cap:
            self.cap.release()
        event.accept()

def main():
    """Run the deepfake detection GUI."""
    app = QApplication(sys.argv)
    window = DeepfakeGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()