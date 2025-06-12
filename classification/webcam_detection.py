import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import mediapipe as mp

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Thiết bị: {device}")

# Tạo model với cùng kiến trúc như lúc training
def create_model():
    """Create and modify Xception model for binary classification."""
    model = timm.create_model("xception", pretrained=False, num_classes=1)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
    )
    return model.to(device)

# Tải mô hình
model = create_model()
model.load_state_dict(torch.load('G:/Hiep/Deepfake_Detection/classification/models/model_remote_full/deepfake_model_final.pt', map_location=device))
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

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình!")
        break

    # Phát hiện khuôn mặt
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
                    label = "Real" if prediction < 0.7 else "Fake"
                    confidence = prediction if label == "Fake" else 1 - prediction

                    # Màu sắc khác nhau cho Real/Fake
                    color = (0, 255, 0) if label == "Real" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                    cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                except Exception as e:
                    print(f"Lỗi xử lý khuôn mặt: {e}")

    cv2.imshow('Deepfake Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()