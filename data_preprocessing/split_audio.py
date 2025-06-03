import os
import shutil

input_dir = "D:/Deepfake_Detection_project/data_preprocessing/frames_audio"
output_dir = "D:/Deepfake_Detection_project/data_preprocessing/output_audio/"

for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".png"):
            src_path = os.path.join(root, file)
            if "RealAudio" in file:
                split = "train" if hash(file) % 10 < 8 else "test" if hash(file) % 10 == 8 else "validation"
                dst_dir = os.path.join(output_dir, split, "real")
            elif "FakeAudio" in file:
                split = "train" if hash(file) % 10 < 8 else "test" if hash(file) % 10 == 8 else "validation"
                dst_dir = os.path.join(output_dir, split, "fake")
            else:
                continue
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src_path, os.path.join(dst_dir, file))