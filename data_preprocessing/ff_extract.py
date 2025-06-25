import cv2
import os
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

# Configuration
BASE_DIR = "D:/Deepfake_Detection_project/data/FaceForensics_c23/"
OUTPUT_DIR = "/data_preprocessing/ff_frames/"
MAX_FRAMES = 15
TARGET_SIZE = (299, 299)

# Global metadata storage
FRAME_METADATA = []

def extract_frames(args):
    """Extract and save face ff_frames from a video."""
    video_path, output_dir, label, source_name = args
    from mtcnn import MTCNN
    detector = MTCNN()

    # Extract video name from path
    video_name = os.path.basename(video_path).split('.')[0]

    # Create video-specific output directory with label
    video_dir = os.path.join(output_dir, f"{label}_{source_name}_{video_name}")
    print(f"Creating directory: {video_dir}")  # Debug log
    os.makedirs(video_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Cannot read video: {video_path}")
        return

    step = max(total_frames // MAX_FRAMES, 1)
    count = 0
    frame_idx = 0
    success, frame = cap.read()

    while success and count < MAX_FRAMES:
        if frame_idx % step == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_frame)

            if faces:
                x, y, w, h = faces[0]['box']
                x, y = max(x, 0), max(y, 0)

                # Expand face region by 1.3x
                new_w, new_h = int(w * 1.3), int(h * 1.3)
                center_x, center_y = x + w // 2, y + h // 2
                new_x = max(center_x - new_w // 2, 0)
                new_y = max(center_y - new_h // 2, 0)
                new_x_end = min(new_x + new_w, frame.shape[1])
                new_y_end = min(new_y + new_h, frame.shape[0])
                new_x = max(new_x_end - new_w, 0)
                new_y = max(new_y_end - new_h, 0)

                # Crop and resize face
                face_img = frame[new_y:new_y_end, new_x:new_x_end]
                face_img = cv2.resize(face_img, TARGET_SIZE)

                # Save frame
                frame_name = f"frame{count}.jpg"
                output_path = os.path.join(video_dir, frame_name)
                cv2.imwrite(output_path, face_img)

                # Store metadata
                FRAME_METADATA.append({
                    "image_path": output_path,
                    "label": label,
                    "video_source": video_path,
                    "source_folder": source_name
                })
                count += 1

        success, frame = cap.read()
        frame_idx += 1
    cap.release()

def process_videos(video_dir, label, output_dir):
    """Process all videos in a directory using multiprocessing."""
    if not os.path.exists(video_dir):
        print(f"Directory not found: {video_dir}")
        return

    source_name = os.path.basename(os.path.dirname(os.path.dirname(video_dir))).lstrip('_').split('_')[0]
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_args = [(os.path.join(video_dir, v), output_dir, label, source_name) for v in videos]

    with Pool() as pool:
        list(tqdm(pool.imap(extract_frames, video_args), total=len(video_args), desc=f"Processing {label} videos"))

def main():
    """Extract ff_frames from real and fake videos."""
    # Clear old content in OUTPUT_DIR
    if os.path.exists(OUTPUT_DIR):
        for item in os.listdir(OUTPUT_DIR):
            item_path = os.path.join(OUTPUT_DIR, item)
            if os.path.isdir(item_path):
                import shutil
                shutil.rmtree(item_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process real videos
    original_dir = os.path.join(BASE_DIR, "original_sequences")
    if os.path.exists(original_dir):
        for subdir in os.listdir(original_dir):
            if os.path.isdir(os.path.join(original_dir, subdir)):
                video_dir = os.path.join(original_dir, subdir, "c23", "videos")
                if os.path.exists(video_dir):
                    process_videos(video_dir, "real", OUTPUT_DIR)

    # Process fake videos
    manipulated_dir = os.path.join(BASE_DIR, "manipulated_sequences")
    if os.path.exists(manipulated_dir):
        for subdir in os.listdir(manipulated_dir):
            if os.path.isdir(os.path.join(manipulated_dir, subdir)):
                video_dir = os.path.join(manipulated_dir, subdir, "c23", "videos")
                if os.path.exists(video_dir):
                    process_videos(video_dir, "fake", OUTPUT_DIR)

    print("Frame extraction completed!")

if __name__ == "__main__":
    main()