import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp

# Tắt oneDNN để tránh thông báo không cần thiết
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Tải model XceptionNet
model = load_model('D:/project/deepfake_model.h5')  # Thay bằng đường dẫn đến file model của bạn

# Khởi tạo Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Hàm tiền xử lý frame/khuôn mặt
def preprocess_frame(frame, target_size=(299, 299)):
    frame = cv2.resize(frame, target_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = img_to_array(frame) / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Hàm dự đoán deepfake
def predict_deepfake(frame, model):
    processed_frame = preprocess_frame(frame)
    pred = model.predict(processed_frame)
    return pred[0][0]  # Xác suất là fake

# Mở webcam
cap = cv2.VideoCapture(0)  # 0 là webcam mặc định

if not cap.isOpened():
    print("Không thể mở webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ webcam!")
        break

    # Chuyển frame sang RGB cho Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # Kiểm tra nếu phát hiện khuôn mặt
    if results.detections:
        for detection in results.detections:
            # Lấy bounding box của khuôn mặt
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)

            # Cắt khuôn mặt từ frame
            face = frame[max(0, y):y + height, max(0, x):x + width]

            # Kiểm tra xem khuôn mặt có hợp lệ không
            if face.size > 0:
                # Dự đoán deepfake
                prediction = predict_deepfake(face, model)
                label = "Real" if prediction < 0.5 else "Fake"  # Ngưỡng 0.3
                confidence = prediction if label == "Fake" else 1 - prediction

                # Vẽ khung và nhãn lên frame
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow('Deepfake Detection', frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()