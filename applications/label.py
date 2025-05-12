import os
import pandas as pd

output_dir = "D:/Deepfake_Detection_project/frames_mtcnn/"
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
df.to_csv("D:/Deepfake_Detection_project/labels_mtcnn.csv", index=False)
print("Đã tạo file labels.csv!")