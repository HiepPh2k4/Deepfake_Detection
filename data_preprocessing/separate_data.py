import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc file nhãn
df = pd.read_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/labels_mtcnn.csv")

# Chia dữ liệu
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

# Lưu thành các file riêng
train_df.to_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/train_labels_mtcnn.csv", index=False)
val_df.to_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/val_labels_mtcnn.csv", index=False)
test_df.to_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/test_labels_mtcnn.csv", index=False)
print("Đã chia dữ liệu thành công!")