import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Paths to the directories containing extracted frames and original videos
output_dir = "/data_preprocessing/frames_mtcnn/"
faceforensics_real_dir = "/data/FaceForensics_c23/original_sequences/"
faceforensics_fake_dir = "/data/FaceForensics_c23/manipulated_sequences/"

# Function to count the number of videos and attempted frames
def count_videos_and_attempts(video_dir, max_frames=10):
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    num_videos = len(videos)
    total_attempts = num_videos * max_frames  # Each video attempts up to max_frames frames
    return num_videos, total_attempts

# Function to count saved frames and build y_true, y_pred
def calculate_detection_results():
    # Count videos and attempted frames
    real_videos, real_attempts = count_videos_and_attempts(faceforensics_real_dir)
    fake_videos, fake_attempts = count_videos_and_attempts(faceforensics_fake_dir)
    total_videos = real_videos + fake_videos
    total_attempts = real_attempts + fake_attempts

    # Count saved frames (y_pred = 1)
    saved_frames = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])

    # Print summary
    print(f"Total videos processed: {total_videos} (Real: {real_videos}, Fake: {fake_videos})")
    print(f"Total frames attempted for extraction: {total_attempts}")
    print(f"Total frames saved (faces detected): {saved_frames}")

    # Build y_true and y_pred
    y_true = [1] * total_attempts  # Assume all selected frames should have faces
    y_pred = [1] * saved_frames + [0] * (total_attempts - saved_frames)  # 1 for saved frames, 0 for undetected

    return y_true, y_pred

# Function to calculate confusion matrix and metrics
def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        print("No data available to calculate metrics.")
        return

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (2x2):")
    print("[[True Negative  False Positive]")
    print(" [False Negative True Positive]]")
    print(cm)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    # Calculate y_true and y_pred from saved frames
    y_true, y_pred = calculate_detection_results()

    # Calculate and print metrics
    calculate_metrics(y_true, y_pred)

    print("Finished calculating metrics from extracted frames!")