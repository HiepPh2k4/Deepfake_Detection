import pandas as pd

# train_df = pd.read_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full/train_rgb.csv")
# val_df = pd.read_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full/val_rgb.csv")
# test_df = pd.read_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_full/test_rgb.csv")

# train_df = pd.read_csv("D:/Study/USTH_B3_ICT_GEN_13/Semester-2/Deepfake_Detection/data_preprocessing/output_split/label_split_500/train_rgb.csv")
# val_df = pd.read_csv("D:/Study/USTH_B3_ICT_GEN_13/Semester-2/Deepfake_Detection/data_preprocessing/output_split/label_split_500/val_rgb.csv")
# test_df = pd.read_csv("D:/Study/USTH_B3_ICT_GEN_13/Semester-2/Deepfake_Detection/data_preprocessing/output_split/label_split_500/test_rgb.csv")

train_df = pd.read_csv("G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_test_remote/train.csv")
val_df = pd.read_csv("G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_test_remote/val.csv")
test_df = pd.read_csv("G:/Hiep/Deepfake_Detection/data_preprocessing/output_split/label_test_remote/test.csv")

print("Train:", train_df["label"].value_counts())
print("Validation:", val_df["label"].value_counts())
print("Test:", test_df["label"].value_counts())