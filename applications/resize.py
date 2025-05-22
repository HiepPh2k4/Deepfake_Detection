import cv2
import numpy as np
from tensorflow.keras.applications.xception import preprocess_input


def load_and_preprocess_image(image_path):
    # Đọc hình ảnh
    img = cv2.imread(image_path)
<<<<<<< Updated upstream:applications/resize.py
    img = cv2.resize(img, (299, 299))  # Kích thước XceptionNet
    img = preprocess_input(img)  # Chuẩn hóa cho XceptionNet
    return img

# Tạo generator để tiết kiệm bộ nhớ
def data_generator(df, batch_size=32):
=======
    if img is None:
        raise ValueError(f"Không thể đọc hình ảnh từ {image_path}")

    # Đảm bảo kích thước là 299x299 (đã được xử lý trong bước trích xuất)
    if img.shape[:2] != (299, 299):
        img = cv2.resize(img, (299, 299))

    # Chuẩn hóa hình ảnh theo chuẩn của XceptionNet
    img = preprocess_input(img)  # Xception-specific normalization
    return img


def data_generator(df, batch_size=16):
>>>>>>> Stashed changes:classification/resize.py
    while True:
        for start in range(0, len(df), batch_size):
            batch_df = df[start:start + batch_size]
            images = []
            labels = []
            for _, row in batch_df.iterrows():
<<<<<<< Updated upstream:applications/resize.py
                img = load_and_preprocess_image(row["image_path"])
                images.append(img)
                label = 1 if row["label"] == "real" else 0
                labels.append(label)
            yield np.array(images), np.array(labels)
=======
                try:
                    img = load_and_preprocess_image(row["image_path"])
                    images.append(img)
                    label = 0 if row["label"] == "real" else 1  # 0: real, 1: fake
                    labels.append(label)
                except Exception as e:
                    print(f"Lỗi khi xử lý {row['image_path']}: {e}")
                    continue
            if len(images) > 0:
                yield np.array(images), np.array(labels)
>>>>>>> Stashed changes:classification/resize.py
