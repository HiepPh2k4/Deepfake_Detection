import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import mediapipe as mp

# Load the XceptionNet model
model = load_model('D:/Study/USTH_B3_ICT_GEN_13/Semester-2/Deepfake_Detection/classification/models/deepfake_model1.h5')

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Function to preprocess the frame/face
def preprocess_frame(frame, target_size=(299, 299)):
    frame = cv2.resize(frame, target_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = img_to_array(frame)
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Function to predict deepfake on a video
def predict_video(video_path, model, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert frame to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            # Check if faces are detected
            if results.detections:
                for detection in results.detections:
                    # Get the bounding box of the face
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    width, height = int(bbox.width * w), int(bbox.height * h)

                    # Crop the face from the frame
                    face = frame[max(0, y):y + height, max(0, x):x + width]

                    # Check if the face region is valid
                    if face.size > 0:
                        processed_frame = preprocess_frame(face)
                        pred = model.predict(processed_frame)
                        predictions.append(pred[0][0])
                        print(f"Frame {frame_count}: Prediction = {pred[0][0]:.4f}")
                    break  # Only process the first detected face

        frame_count += 1

    cap.release()

    if predictions:
        max_prediction = np.max(predictions)  # Use maximum prediction
        print(f"All predictions: {predictions}")
        label = "Real" if max_prediction < 0.5 else "Fake"  # Adjusted threshold
        confidence = max_prediction if label == "Fake" else 1 - max_prediction
        return label, confidence
    else:
        return "Fake", 0

# Test a video
#video_path = 'D:/Deepfake_Detection_project/data/FaceForensics_c23/manipulated_sequences/DeepFakeDetection/c23/videos/02_14__podium_speech_happy__3IUBEKCT.mp4'
video_path = 'D:/Study/USTH_B3_ICT_GEN_13/Semester-2/Deepfake_Detection/real_test_video/1.mp4'
label, confidence = predict_video(video_path, model)
print(f"Video is predicted as: {label}")
print(f"Confidence: {confidence:.2f}")