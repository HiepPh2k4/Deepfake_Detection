import os
import pandas as pd

output_dir = "D:/project/frames/"
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
df.to_csv("D:/project/labels.csv", index=False)
print("Đã tạo file labels.csv!")