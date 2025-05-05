import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import applications, models, layers

Xception = applications.Xception
Model = models.Model
Dense = layers.Dense
GlobalAveragePooling2D = layers.GlobalAveragePooling2D
from resize import data_generator  # Nhập data_generator từ resize.py

# 1. Đọc dữ liệu đã chia
train_df = pd.read_csv("D:/project/train_labels.csv")
val_df = pd.read_csv("D:/project/val_labels.csv")
test_df = pd.read_csv("D:/project/test_labels.csv")

print(f"Số ảnh train: {len(train_df)}")
print(f"Số ảnh validation: {len(val_df)}")
print(f"Số ảnh test: {len(test_df)}")

# 2. Chuẩn bị mô hình XceptionNet
base_model = Xception(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Đóng băng các tầng của XceptionNet
for layer in base_model.layers:
    layer.trainable = False

# Biên dịch mô hình
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 3. Huấn luyện mô hình
batch_size = 32
train_generator = data_generator(train_df, batch_size)
val_generator = data_generator(val_df, batch_size)

model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // batch_size,
    validation_data=val_generator,
    validation_steps=len(val_df) // batch_size,
    epochs=10
)

# 4. Lưu mô hình
model.save("D:/project/deepfake_model.h5")
print("Đã huấn luyện và lưu mô hình!")

# 5. Đánh giá trên tập test
test_generator = data_generator(test_df, batch_size)
steps = len(test_df) // batch_size
loss, accuracy = model.evaluate(test_generator, steps=steps)
print(f"Độ chính xác trên tập test: {accuracy * 100:.2f}%")