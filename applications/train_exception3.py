import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import applications, models, layers
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Aliases for clarity
Xception = applications.Xception
Model = models.Model
Dense = layers.Dense
GlobalAveragePooling2D = layers.GlobalAveragePooling2D
from resize import data_generator  # Import data_generator from resize.py

# 1. Load the data
train_df = pd.read_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/train_labels_mtcnn3.csv")
val_df = pd.read_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/val_labels_mtcnn3.csv")
test_df = pd.read_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/test_labels_mtcnn3.csv")

print(f"Number of training images: {len(train_df)}")
print(f"Number of validation images: {len(val_df)}")
print(f"Number of test images: {len(test_df)}")

# 2. Prepare the XceptionNet model
base_model = Xception(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the XceptionNet layers for the first 3 epochs
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with specified metrics and learning rate from the paper
model.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        Precision(name="precision"),
        Recall(name="recall"),
        AUC(name="auc")
    ]
)

# 3. Train the model
batch_size = 32
train_generator = data_generator(train_df, batch_size)
val_generator = data_generator(val_df, batch_size)

# Apply class weighting for the real:fake imbalance (1:4 as per the paper)
class_weight = {0: 4.0, 1: 1.0}  # 0: real, 1: fake

# Train the model for 3 epochs as per the paper's pre-training phase
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // batch_size,
    validation_data=val_generator,
    validation_steps=len(val_df) // batch_size,
    epochs=3,
    class_weight=class_weight,
    verbose=1  # Ensures detailed output during training
)

# 4. Display detailed training results for each epoch
print("\nDetailed Training Results:")
for epoch in range(len(history.history['loss'])):
    print(f"Epoch {epoch + 1}/{len(history.history['loss'])}:")
    print(f"  Train Loss: {history.history['loss'][epoch]:.4f}")
    print(f"  Train Accuracy: {history.history['accuracy'][epoch]:.4f}")
    print(f"  Train Precision: {history.history['precision'][epoch]:.4f}")
    print(f"  Train Recall: {history.history['recall'][epoch]:.4f}")
    print(f"  Train AUC: {history.history['auc'][epoch]:.4f}")
    print(f"  Val Loss: {history.history['val_loss'][epoch]:.4f}")
    print(f"  Val Accuracy: {history.history['val_accuracy'][epoch]:.4f}")
    print(f"  Val Precision: {history.history['val_precision'][epoch]:.4f}")
    print(f"  Val Recall: {history.history['val_recall'][epoch]:.4f}")
    print(f"  Val AUC: {history.history['val_auc'][epoch]:.4f}")

# 5. Save the trained model
model.save("D:/Deepfake_Detection_project/classification/models/deepfake_model_mtcnn3.h5")
print("Model trained and saved!")

# 6. Evaluate the model on the test set
test_generator = data_generator(test_df, batch_size)
steps = len(test_df) // batch_size
loss, accuracy, precision, recall, auc = model.evaluate(test_generator, steps=steps)
print(f"Test set accuracy: {accuracy * 100:.2f}%")
print(f"Test set precision: {precision * 100:.2f}%")
print(f"Test set recall: {recall * 100:.2f}%")
print(f"Test set AUC: {auc * 100:.2f}%")

# Calculate the F1-score manually
y_true = []
y_pred = []
for _ in range(steps):
    images, labels = next(test_generator)
    preds = model.predict(images)
    y_true.extend(labels.flatten())
    y_pred.extend((preds > 0.5).astype(int).flatten())

f1 = f1_score(y_true, y_pred)
print(f"Test set F1-score: {f1 * 100:.2f}%")

# 7. Plot the Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.title("Confusion Matrix on Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("D:/Deepfake_Detection_project/classification/output/confusion_matrix3.png")
plt.show()

# 8. Plot the training history
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

plt.savefig("D:/Deepfake_Detection_project/classification/output/training_plots3.png")
plt.show()