import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import timm
import mediapipe as mp

# Kiểm tra nhãn
def check_label_consistency():
    print("Label: 0 = Real, 1 = Fake")
    print("Model: Xception, num_classes=1, BCEWithLogitsLoss")
check_label_consistency()

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Thiết bị: {device}")

# Tải mô hình
model = timm.create_model('xception', pretrained=False, num_classes=1).to(device)
model.load_state_dict(torch.load('D:/Deepfake_Detection_project/classification/models/models_full/deepfake_model_best.pt'))
model.eval()
print("Đã tải mô hình XceptionNet!")

# Khởi tạo Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Chuẩn hóa frame
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_and_predict(frame, model):
    """Chuẩn hóa và dự đoán deepfake cho frame."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(frame)
        pred = torch.sigmoid(output).item()
    return pred

def predict_video(video_path, model, frame_interval=10):
    """Dự đoán deepfake trên video, trả về nhãn và xác suất."""
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
                    # Mở rộng vùng khuôn mặt 1.3 lần
                    x, y = int(bbox.xmin * w - 0.15 * bbox.width * w), int(bbox.ymin * h - 0.15 * bbox.height * h)
                    width, height = int(bbox.width * w * 1.3), int(bbox.height * h * 1.3)
                    x, y = max(0, x), max(0, y)
                    width, height = min(w - x, width), min(h - y, height)

                    face = frame[y:y + height, x:x + width]
                    if face.size > 0 and face.shape[0] >= 50 and face.shape[1] >= 50:
                        try:
                            prediction = preprocess_and_predict(face, model)
                            predictions.append(prediction)
                            print(f"Frame {frame_count}: Prediction = {prediction:.4f}")
                        except Exception as e:
                            print(f"Lỗi xử lý khung hình {frame_count}: {e}")
                    break  # Chỉ xử lý khuôn mặt đầu tiên

        frame_count += 1

    cap.release()

    if predictions:
        max_prediction = np.max(predictions)
        label = "Real" if max_prediction < 0.7 else "Fake"
        confidence = max_prediction if label == "Fake" else 1 - max_prediction
        return label, confidence
    return "Fake", 0

# Kiểm tra video
# video_path = 'D:/Deepfake_Detection_project/data/FaceForensics_c23/original_sequences/actors/c23/videos/01__walk_down_hall_angry.mp4'
video_path = 'D:/Deepfake_Detection_project/data/FaceForensics_c23/manipulated_sequences/FaceSwap/c23/videos/014_790.mp4'
label, confidence = predict_video(video_path, model)
print(f"Video được dự đoán là: {label}")
print(f"Xác suất: {confidence:.2f}")