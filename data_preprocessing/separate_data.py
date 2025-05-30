import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc file nhãn
# df = pd.read_csv("G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_split_remote/label_rgb.csv")
df = pd.read_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full/label_rgb.csv")
# Chia dữ liệu
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

# Lưu thành các file riêng
# train_df.to_csv("G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_split_remote/train_rgb.csv", index=False)
# val_df.to_csv("G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_split_remote/val_rgb.csv", index=False)
# test_df.to_csv("G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_split_remote/test_rgb.csv", index=False)

train_df.to_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full/train_rgb.csv", index=False)
val_df.to_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full/val_rgb.csv", index=False)
test_df.to_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full/test_rgb.csv", index=False)

print("completely!")