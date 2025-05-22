import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
import mediapipe as mp

# Disable oneDNN to avoid unnecessary warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the XceptionNet model
model = load_model('/classification/models/model_full/deepfake_model_pretrain_3.h5')

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to preprocess the frame/face
def preprocess_frame(frame, target_size=(299, 299)):
    frame = cv2.resize(frame, target_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = preprocess_input(frame)  # Use Xception-specific preprocessing
    frame = np.expand_dims(frame, axis=0)
    return frame

# Function to predict deepfake
def predict_deepfake(frame, model):
    processed_frame = preprocess_frame(frame)
    pred = model.predict(processed_frame)
    return pred[0][0]  # Probability of being fake

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

if not cap.isOpened():
    print("Unable to open webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to read frame from webcam!")
        break

    # Convert the frame to RGB for Mediapipe
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
                prediction = predict_deepfake(face, model)
                label = "Fake" if prediction >= 0.75 else "Real"
                confidence = prediction if label == "Fake" else 1 - prediction

                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Deepfake Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()