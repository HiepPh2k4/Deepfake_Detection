import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.applications.xception import preprocess_input

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (299, 299))  # Kích thước XceptionNet
    img = preprocess_input(img)  # Chuẩn hóa cho XceptionNet
    return img

# Tạo generator để tiết kiệm bộ nhớ
def data_generator(df, batch_size=32):
    while True:
        for start in range(0, len(df), batch_size):
            batch_df = df[start:start + batch_size]
            images = []
            labels = []
            for _, row in batch_df.iterrows():
                img = load_and_preprocess_image(row["image_path"])
                images.append(img)
                label = 1 if row["label"] == "real" else 0
                labels.append(label)
            yield np.array(images), np.array(labels)