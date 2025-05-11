import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

model = load_model('D:/Deepfake_Detection_project/deepfake_model.h5')

# Hàm tiền xử lý frame
def preprocess_frame(frame, target_size=(299, 299)):
    frame = cv2.resize(frame, target_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = img_to_array(frame)
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def predict_video(video_path, model, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            processed_frame = preprocess_frame(frame)
            pred = model.predict(processed_frame)
            predictions.append(pred[0][0])

        frame_count += 1

    cap.release()

    if predictions:
        avg_prediction = np.mean(predictions)
        label = "Real" if avg_prediction < 0.3 else "Fake"
        confidence = avg_prediction if label == "Fake" else 1 - avg_prediction
        return label, confidence
    else:
        return "Fake", 0

video_path = 'D:/Deepfake_Detection_project/data/FaceForensics/original_sequences/actors/c40/videos/09__outside_talking_still_laughing.mp4'

label, confidence = predict_video(video_path, model)
print(f"Video is predicted as: {label} \n")
print(f"confidence: {confidence:.2f}")