import os
import pandas as pd

<<<<<<< Updated upstream:applications/label.py
output_dir = "D:/Deepfake_Detection_project/frames2/"
=======
output_dir = "D:/Deepfake_Detection_project/data_preprocessing/frames_mtcnn_rgb_500/"
>>>>>>> Stashed changes:data_preprocessing/label.py
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
<<<<<<< Updated upstream:applications/label.py
df.to_csv("D:/Deepfake_Detection_project/labels2.csv", index=False)
=======
df.to_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_500/label_rgb.csv", index=False)
>>>>>>> Stashed changes:data_preprocessing/label.py
print("Đã tạo file labels.csv!")