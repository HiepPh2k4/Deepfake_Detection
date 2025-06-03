import os
import pandas as pd

output_dir = "G:/Hiep/Deepfake_Detection/data_preprocessing/frames/frames_mtcnn_rgb/"
# output_dir = "/data_preprocessing/frames_full/"
images = os.listdir(output_dir)
data = []

for img in images:
    if img.startswith("real_"):
        label = "real"
    elif img.startswith("fake_"):
        label = "fake"
    else:
        continue
    data.append([os.path.join(output_dir, img), label])

df = pd.DataFrame(data, columns=["image_path", "label"])
# df.to_csv("G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_split_remote/label_rgb.csv", index=False)
df.to_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full/label_rgb.csv", index=False)
print("created labels.csv!")