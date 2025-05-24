import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import timm
import mediapipe as mp

# Định nghĩa thiết bị (GPU hoặc CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Thiết bị: {device}")
# if torch.cuda.is_available():
#     print(f"Phiên bản PyTorch: {torch.__version__}")
#     print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
#     print(f"CUDA Version: {torch.version.cuda}")

device = torch.device("cpu")

# Tải mô hình XceptionNet
model = timm.create_model('xception', pretrained=False, num_classes=1).to(device)
try:
    #model.load_state_dict(torch.load('G:/Hiep/Deepfake_Detection/classification/models/models_remote_full/deepfake_model_final_18.pt'))
    model.load_state_dict(torch.load('D:/Study/USTH_B3_ICT_GEN_13/Semester-2/Deepfake_Detection/classification/models/models_remote_full/deepfake_model_final_18.pt', map_location=device))
    model.eval()
    print("Đã tải mô hình XceptionNet!")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit()

# Khởi tạo Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

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

# Hàm dự đoán deepfake
def predict_deepfake(frame, model):
    processed_frame = preprocess_frame(frame)
    with torch.no_grad():
        output = model(processed_frame)
        pred = torch.sigmoid(output).item()  # Xác suất là fake
    return pred

# Mở webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ webcam!")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            # Mở rộng vùng khuôn mặt với hệ số 1.3
            x, y = int(bbox.xmin * w - 0.2 * bbox.width * w), int(bbox.ymin * h - 0.2 * bbox.height * h)
            width, height = int(bbox.width * w * 1.3), int(bbox.height * h * 1.3)
            x, y = max(0, x), max(0, y)
            width, height = min(w - x, width), min(h - y, height)

            face = frame[y:y + height, x:x + width]

            if face.size > 0 and face.shape[0] >= 50 and face.shape[1] >= 50:  # Kiểm tra kích thước
                try:
                    # Điều chỉnh ánh sáng/tương phản
                    face = cv2.convertScaleAbs(face, alpha=1.2, beta=20)
                    prediction = predict_deepfake(face, model)
                    label = "Fake" if prediction >= 0.5 else "Real"
                    confidence = prediction if label == "Fake" else 1 - prediction

                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Lỗi khi xử lý khuôn mặt: {e}")

    cv2.imshow('Deepfake Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()