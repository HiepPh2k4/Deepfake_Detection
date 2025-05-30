import os
import cv2
import librosa
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# Đường dẫn gốc của dataset
root_dir = "D:/Deepfake_Detection_project/data/FAKEAVCeleb-v1.2"
output_dir = "D:/Deepfake_Detection_project/data_preprocessing/frames_audio"
os.makedirs(output_dir, exist_ok=True)

# Hàm chuyển audio thành spectrogram
def audio_to_spectrogram(video_path, output_path, sr=22050, n_mels=128):
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)
    video.close()

    y, sr = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.99, 2.99))
    plt.axis('off')
    plt.imshow(S_db, cmap='viridis')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = cv2.imread(output_path)
    img = cv2.resize(img, (299, 299))
    cv2.imwrite(output_path, img)

    os.remove(audio_path)


# Các thư mục chính
main_folders = [
    "FakeVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "RealVideo-FakeAudio",
    "RealVideo-RealAudio"
]

# Duyệt qua tất cả thư mục
for main_folder in main_folders:
    main_path = os.path.join(root_dir, main_folder)
    if not os.path.exists(main_path):
        continue
    for race in ["African", "Asian", "Caucasian"]:
        race_path = os.path.join(main_path, race)
        if not os.path.exists(race_path):
            continue
        for gender in ["men", "women"]:
            gender_path = os.path.join(race_path, gender)
            if not os.path.exists(gender_path):
                continue
            # Duyệt qua các thư mục ID
            for id_folder in os.listdir(gender_path):
                id_path = os.path.join(gender_path, id_folder)
                if not os.path.isdir(id_path):
                    continue
                for filename in os.listdir(id_path):
                    if filename.endswith(".mp4"):
                        video_path = os.path.join(id_path, filename)
                        # Tạo tên thư mục và file: main_folder_race_gender_id_filename
                        out_subdir = os.path.join(output_dir, f"{main_folder}_{race}_{gender}_{id_folder}")
                        os.makedirs(out_subdir, exist_ok=True)
                        output_filename = f"{main_folder}_{race}_{gender}_{id_folder}_{filename[:-4]}.png"
                        output_path = os.path.join(out_subdir, output_filename)
                        audio_to_spectrogram(video_path, output_path)

print("Xử lý toàn bộ dataset FakeAVCeleb hoàn tất!")