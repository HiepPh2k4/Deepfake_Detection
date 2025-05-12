import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import applications, models, layers
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

Xception = applications.Xception
Model = models.Model
Dense = layers.Dense
GlobalAveragePooling2D = layers.GlobalAveragePooling2D
from resize import data_generator  # Import data_generator from resize.py

# 1. Load data
train_df = pd.read_csv("D:/Deepfake_Detection_project/train_labels_mtcnn.csv")
val_df = pd.read_csv("D:/Deepfake_Detection_project/val_labels_mtcnn.csv")
test_df = pd.read_csv("D:/Deepfake_Detection_project/test_labels_mtcnn.csv")

print(f"Number of training images: {len(train_df)}")
print(f"Number of validation images: {len(val_df)}")
print(f"Number of test images: {len(test_df)}")

# 2. Prepare XceptionNet model
base_model = Xception(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze XceptionNet layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model with additional metrics
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        Precision(name="precision"),
        Recall(name="recall"),
        AUC(name="auc")
    ]
)

# 3. Train model
batch_size = 16
train_generator = data_generator(train_df, batch_size)
val_generator = data_generator(val_df, batch_size)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // batch_size,
    validation_data=val_generator,
    validation_steps=len(val_df) // batch_size,
    epochs=3
)

# 4. Save model
model.save("D:/Deepfake_Detection_project/deepfake_model_mtcnn.h5")
print("Model trained and saved!")

# 5. Evaluate on test set
test_generator = data_generator(test_df, batch_size)
steps = len(test_df) // batch_size
loss, accuracy, precision, recall, auc = model.evaluate(test_generator, steps=steps)
print(f"Test set accuracy: {accuracy * 100:.2f}%")
print(f"Test set precision: {precision * 100:.2f}%")
print(f"Test set recall: {recall * 100:.2f}%")
print(f"Test set AUC: {auc * 100:.2f}%")

# Calculate F1-score manually
y_true = []
y_pred = []
for _ in range(steps):
    images, labels = next(test_generator)
    preds = model.predict(images)
    y_true.extend(labels.flatten())
    y_pred.extend((preds > 0.5).astype(int).flatten())

f1 = f1_score(y_true, y_pred)
print(f"Test set F1-score: {f1 * 100:.2f}%")

# 6. Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Deepfake"], yticklabels=["Real", "Deepfake"])
plt.title("Confusion Matrix on Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("D:/Deepfake_Detection_project/confusion_matrix.png")
plt.show()

# 7. Plot training history
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig("D:/Deepfake_Detection_project/training_plots.png")
plt.show()