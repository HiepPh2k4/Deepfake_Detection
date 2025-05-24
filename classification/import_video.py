import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import timm
import mediapipe as mp

# Định nghĩa thiết bị (GPU hoặc CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Thiết bị: {device}")
if torch.cuda.is_available():
    print(f"Phiên bản PyTorch: {torch.__version__}")
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Tải mô hình XceptionNet
model = timm.create_model('xception', pretrained=False, num_classes=1).to(device)
try:
    model.load_state_dict(torch.load('G:/Hiep/Deepfake_Detection/classification/models/models_remote/deepfake_model_final_18.pt'))
    model.eval()
    print("Đã tải mô hình XceptionNet!")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit()

# Khởi tạo Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Hàm chuẩn hóa khung hình/khuôn mặt
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = transform(frame)
    frame = frame.unsqueeze(0).to(device)
    return frame

# Hàm dự đoán deepfake trên video
def predict_video(video_path, model, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y = int(bbox.xmin * w - 0.2 * bbox.width * w), int(bbox.ymin * h - 0.2 * bbox.height * h)
                    width, height = int(bbox.width * w * 1.3), int(bbox.height * h * 1.3)
                    x, y = max(0, x), max(0, y)
                    width, height = min(w - x, width), min(h - y, height)

                    face = frame[y:y + height, x:x + width]

                    if face.size > 0 and face.shape[0] >= 50 and face.shape[1] >= 50:
                        try:
                            face = cv2.convertScaleAbs(face, alpha=1.2, beta=20)  # Điều chỉnh ánh sáng
                            prediction = predict_deepfake(face, model)
                            predictions.append(prediction)
                            print(f"Frame {frame_count}: Prediction = {prediction:.4f}")
                        except Exception as e:
                            print(f"Lỗi khi xử lý khung hình {frame_count}: {e}")
                    break  # Chỉ xử lý khuôn mặt đầu tiên

        frame_count += 1

    cap.release()

    if predictions:
        max_prediction = np.max(predictions)  # Lấy xác suất tối đa
        print(f"Tất cả dự đoán: {predictions}")
        label = "Real" if max_prediction < 0.5 else "Fake"
        confidence = max_prediction if label == "Fake" else 1 - max_prediction
        return label, confidence
    else:
        return "Fake", 0

# Hàm dự đoán deepfake trên một khung hình
def predict_deepfake(frame, model):
    processed_frame = preprocess_frame(frame)
    with torch.no_grad():
        output = model(processed_frame)
        pred = torch.sigmoid(output).item()  # Xác suất là fake
    return pred

# Kiểm tra video
video_path = 'G:/Hiep/Deepfake_Detection/data/FaceForensics_c23/original_sequences/youtube/c23/videos/720.mp4'
label, confidence = predict_video(video_path, model)
print(f"Video được dự đoán là: {label}")
print(f"Xác suất: {confidence:.2f}")