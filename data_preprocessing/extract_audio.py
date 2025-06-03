import os
import cv2
import librosa
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from multiprocessing import Pool
from tqdm import tqdm

# Đường dẫn gốc của dataset
root_dir = "D:/Deepfake_Detection_project/data/FakeAVCeleb_v1.2"
output_dir = "D:/Deepfake_Detection_project/data_preprocessing/frames_audio"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")

# Hàm chuyển audio thành spectrogram
def audio_to_spectrogram(args):
    video_path, output_path, sr, n_mels = args
    print(f"Processing video: {video_path}")
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist: {video_path}")
        return
    try:
        video = VideoFileClip(video_path)
        audio_path = f"temp_audio_{os.getpid()}.wav"  # Đặt tên file tạm theo PID để tránh xung đột
        video.audio.write_audiofile(audio_path)
        video.close()
        print(f"Audio extracted to: {audio_path}")

        y, sr = librosa.load(audio_path, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(2.99, 2.99))
        plt.axis('off')
        plt.imshow(S_db, cmap='viridis')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Spectrogram saved to: {output_path}")

        img = cv2.imread(output_path)
        if img is None:
            print(f"Error: Failed to load spectrogram image: {output_path}")
            return
        img = cv2.resize(img, (299, 299))
        cv2.imwrite(output_path, img)
        print(f"Resized spectrogram saved to: {output_path}")

        os.remove(audio_path)
        print(f"Temporary audio file removed: {audio_path}")
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")

# Thu thập tất cả các file video để xử lý
video_tasks = []
main_folders = [
    "FakeVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "RealVideo-FakeAudio",
    "RealVideo-RealAudio"
]

for main_folder in main_folders:
    main_path = os.path.join(root_dir, main_folder)
    print(f"Checking folder: {main_path}")
    if not os.path.exists(main_path):
        print(f"Folder not found: {main_path}")
        continue
    for race in ["African", "Asian_East", "Asian_South", "Caucasian_American", "Caucasian_European"]:
        race_path = os.path.join(main_path, race)
        if not os.path.exists(race_path):
            print(f"Folder not found: {race_path}")
            continue
        for gender in ["men", "women"]:
            gender_path = os.path.join(race_path, gender)
            if not os.path.exists(gender_path):
                print(f"Folder not found: {gender_path}")
                continue
            for id_folder in os.listdir(gender_path):
                id_path = os.path.join(gender_path, id_folder)
                if not os.path.isdir(id_path):
                    continue
                print(f"Scanning ID folder: {id_path}")
                mp4_files = [f for f in os.listdir(id_path) if f.endswith(".mp4")]
                if not mp4_files:
                    print(f"No .mp4 files found in {id_path}")
                for filename in mp4_files:
                    video_path = os.path.join(id_path, filename)
                    out_subdir = os.path.join(output_dir, f"{main_folder}_{race}_{gender}_{id_folder}")
                    os.makedirs(out_subdir, exist_ok=True)
                    output_filename = f"{main_folder}_{race}_{gender}_{id_folder}_{filename[:-4]}.png"
                    output_path = os.path.join(out_subdir, output_filename)
                    video_tasks.append((video_path, output_path, 22050, 128))

# Sử dụng multiprocessing và tqdm để xử lý song song
if __name__ == "__main__":
    print(f"Total videos to process: {len(video_tasks)}")
    if video_tasks:
        with Pool() as pool:
            list(tqdm(pool.imap(audio_to_spectrogram, video_tasks), total=len(video_tasks), desc="Processing videos"))
    else:
        print("No videos found to process.")

print("Xử lý toàn bộ dataset FakeAVCeleb hoàn tất!")