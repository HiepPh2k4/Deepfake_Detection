import os
import cv2
import random
import glob
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
from mtcnn import MTCNN

# Configuration
DATA_DIR = "D:/Deepfake_Detection_project/data/FaceForensics_c23/"
OUTPUT_DIR = "D:/Deepfake_Detection_project/data_preprocessing/frames/youtube/"
EXCLUDED_FOLDERS = ["actors", "DeepFakeDetection", "FaceShifter"]
MAX_FRAMES_PER_VIDEO = 20
TARGET_FRAME_SIZE = (299, 299)
TARGET_VIDEO_COUNT = 720

def setup_output_directory():
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def extract_frames(video_path, output_folder, label, source_folder_name, max_frames=MAX_FRAMES_PER_VIDEO, target_size=TARGET_FRAME_SIZE):
    """Extract frames from a video, detect faces, crop them, and save the results."""
    detector = MTCNN()

    print(f"Processing video: {video_path}")

    # Open video and validate
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Video {video_path} is invalid or empty.")
        cap.release()
        return

    # Calculate frame indices to extract
    step = max(total_frames // max_frames, 1)
    frame_indices = [i * step for i in range(max_frames)]

    # Process each frame
    for idx, frame_idx in enumerate(frame_indices):
        # Seek to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_idx} from {video_path}, skipping.")
            continue

        # Convert to RGB for face detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(frame_rgb)

        if faces:
            # Get coordinates of the first detected face
            x, y, w, h = faces[0]['box']
            x, y = max(x, 0), max(y, 0)

            # Expand the crop area
            expansion_factor = 1.3
            new_w = int(w * expansion_factor)
            new_h = int(h * expansion_factor)
            center_x = x + w // 2
            center_y = y + h // 2
            new_x = max(center_x - new_w // 2, 0)
            new_y = max(center_y - new_h // 2, 0)
            new_x_end = min(new_x + new_w, frame.shape[1])
            new_y_end = min(new_y + new_h, frame.shape[0])
            new_x = max(new_x_end - new_w, 0)
            new_y = max(new_y_end - new_h, 0)

            # Crop and resize the face
            face_img = frame[new_y:new_y_end, new_x:new_x_end]
            face_img = cv2.resize(face_img, target_size)

            # Save the processed frame
            frame_name = f"{label}_{source_folder_name}_{os.path.basename(video_path).split('.')[0]}_frame{idx}.jpg"
            output_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(output_path, face_img)
            print(f"Saved frame: {output_path}")
        else:
            print(f"No face detected in frame {frame_idx}, skipping.")

    # Release video capture
    cap.release()

def process_video(args):
    """Wrapper function for multiprocessing to process a single video."""
    video_path, label, output_folder, source_folder_name = args
    extract_frames(video_path, output_folder, label, source_folder_name)

def collect_and_sample_videos(data_dir, target_count=TARGET_VIDEO_COUNT):
    """Collect valid videos and randomly sample the target number."""
    video_files = []
    folder_counts = {}

    # Collect videos from original sequences
    original_dir = os.path.join(data_dir, "original_sequences")
    if os.path.exists(original_dir):
        for subdir in os.listdir(original_dir):
            if subdir in EXCLUDED_FOLDERS or not os.path.isdir(os.path.join(original_dir, subdir)):
                continue
            video_dir = os.path.join(original_dir, subdir, "c23", "videos")
            if os.path.exists(video_dir):
                videos = glob.glob(os.path.join(video_dir, "*.mp4"))
                valid_videos = []
                for video in videos:
                    cap = cv2.VideoCapture(video)
                    if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                        valid_videos.append(video)
                    else:
                        print(f"Invalid video: {video}")
                    cap.release()
                folder_counts[video_dir] = {"count": len(valid_videos), "label": "real"}
                video_files.extend([(v, "real", video_dir) for v in valid_videos])

    # Collect videos from manipulated sequences
    manipulated_dir = os.path.join(data_dir, "manipulated_sequences")
    if os.path.exists(manipulated_dir):
        for subdir in os.listdir(manipulated_dir):
            if subdir in EXCLUDED_FOLDERS or not os.path.isdir(os.path.join(manipulated_dir, subdir)):
                continue
            video_dir = os.path.join(manipulated_dir, subdir, "c23", "videos")
            if os.path.exists(video_dir):
                videos = glob.glob(os.path.join(video_dir, "*.mp4"))
                valid_videos = []
                for video in videos:
                    cap = cv2.VideoCapture(video)
                    if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                        valid_videos.append(video)
                    else:
                        print(f"Invalid video: {video}")
                    cap.release()
                folder_counts[video_dir] = {"count": len(valid_videos), "label": "fake"}
                video_files.extend([(v, "fake", video_dir) for v in valid_videos])

    print(f"Total valid videos found: {len(video_files)}")
    print("Videos per folder:", {k: v["count"] for k, v in folder_counts.items()})

    # Sample videos from each folder
    selected_videos = []
    folders = list(folder_counts.keys())
    videos_per_folder = max(1, target_count // len(folders))

    for folder in folders:
        folder_videos = [(v, label, folder) for v, label, f in video_files if f == folder]
        if folder_videos:
            sampled_videos = random.sample(folder_videos, min(len(folder_videos), videos_per_folder))
            selected_videos.extend(sampled_videos)

    # Add more videos if needed to reach target count
    if len(selected_videos) < target_count:
        remaining_videos = [v for v in video_files if v not in selected_videos]
        additional_videos = random.sample(remaining_videos,
                                         min(len(remaining_videos), target_count - len(selected_videos)))
        selected_videos.extend(additional_videos)

    print(f"Selected {len(selected_videos)} videos from all folders.")
    return selected_videos[:target_count]

def process_videos(selected_videos, output_folder):
    """Process all selected videos using multiprocessing."""
    video_args = [
        (
            video_path,
            label,
            output_folder,
            os.path.basename(os.path.dirname(os.path.dirname(video_dir)))
        )
        for video_path, label, video_dir in selected_videos
    ]

    print(f"Starting processing of {len(video_args)} videos...")
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(process_video, video_args), total=len(video_args), desc="Processing videos"):
            pass

def main():
    """Main function to orchestrate the video processing pipeline."""
    # Set multiprocessing start method for Windows compatibility
    mp.set_start_method("spawn", force=True)

    # Create output directory
    setup_output_directory()

    # Collect and sample videos
    selected_videos = collect_and_sample_videos(DATA_DIR)

    # Process videos
    process_videos(selected_videos, OUTPUT_DIR)

    print("Completed frame extraction!")

if __name__ == '__main__':
    main()