import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.applications.xception import preprocess_input

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (299, 299))  # XceptionNet size
    img = preprocess_input(img)  # Xception-specific normalization
    return img

# Create generator to save memory
def data_generator(df, batch_size=32):
    while True:
        for start in range(0, len(df), batch_size):
            batch_df = df[start:start + batch_size]
            images = []
            labels = []
            for _, row in batch_df.iterrows():
                img = load_and_preprocess_image(row["image_path"])
                images.append(img)
                label = 0 if row["label"] == "real" else 1  # 0: real, 1: fake
                labels.append(label)
            yield np.array(images), np.array(labels)