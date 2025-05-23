import pandas as pd

# train_df = pd.read_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_500/train_rgb.csv")
# val_df = pd.read_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_500/val_rgb.csv")
# test_df = pd.read_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_500/test_rgb.csv")

train_df = pd.read_csv("D:/Study/USTH_B3_ICT_GEN_13/Semester-2/Deepfake_Detection/data_preprocessing/output_split/label_split_500/train_rgb.csv")
val_df = pd.read_csv("D:/Study/USTH_B3_ICT_GEN_13/Semester-2/Deepfake_Detection/data_preprocessing/output_split/label_split_500/val_rgb.csv")
test_df = pd.read_csv("D:/Study/USTH_B3_ICT_GEN_13/Semester-2/Deepfake_Detection/data_preprocessing/output_split/label_split_500/test_rgb.csv")

print("Train:", train_df["label"].value_counts())
print("Validation:", val_df["label"].value_counts())
print("Test:", test_df["label"].value_counts())