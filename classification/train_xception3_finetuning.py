import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import applications, models, layers
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Aliases for clarity
Xception = applications.Xception
Model = models.Model
Dense = layers.Dense
GlobalAveragePooling2D = layers.GlobalAveragePooling2D
from resize import data_generator

# 1. Load the data
train_df = pd.read_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_500/train_rgb.csv")
val_df = pd.read_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_500/val_rgb.csv")
test_df = pd.read_csv("D:/Deepfake_Detection_project/data_preprocessing/output_split/label_split_500/test_rgb.csv")

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

# 3. Freeze the XceptionNet layers for the first 3 epochs (pre-training)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
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

# 4. Train the model for 3 epochs (pre-training)
batch_size = 16
train_generator = data_generator(train_df, batch_size)
val_generator = data_generator(val_df, batch_size)

class_weight = {0: 4.0, 1: 1.0}  # 0: real, 1: fake

history_pretrain = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // batch_size,
    validation_data=val_generator,
    validation_steps=len(val_df) // batch_size,
    epochs=3,
    class_weight=class_weight,
    verbose=1
)

# 5. Save the model after pre-training
model.save("D:/Deepfake_Detection_project/classification/models/model_500/deepfake_model_pretrain_3.h5")
print("Model after pre-training saved!")

# 6. Unfreeze the XceptionNet layers for fine-tuning
for layer in model.layers:
    layer.trainable = True

# Recompile the model
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

# 7. Set up callbacks for fine-tuning
checkpoint = ModelCheckpoint(
    "D:/Deepfake_Detection_project/classification/models/model_500/deepfake_model_best_3.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    mode="max",
    verbose=1,
    restore_best_weights=True
)

# 8. Train the model for 15 epochs (fine-tuning)
history_finetune = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // batch_size,
    validation_data=val_generator,
    validation_steps=len(val_df) // batch_size,
    epochs=15,
    class_weight=class_weight,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# 9. Display detailed training results for fine-tuning
print("\nDetailed Training Results (Fine-Tuning):")
for epoch in range(len(history_finetune.history['loss'])):
    print(f"Epoch {epoch + 1}/{len(history_finetune.history['loss'])}:")
    print(f"  Train Loss: {history_finetune.history['loss'][epoch]:.4f}")
    print(f"  Train Accuracy: {history_finetune.history['accuracy'][epoch]:.4f}")
    print(f"  Train Precision: {history_finetune.history['precision'][epoch]:.4f}")
    print(f"  Train Recall: {history_finetune.history['recall'][epoch]:.4f}")
    print(f"  Train AUC: {history_finetune.history['auc'][epoch]:.4f}")
    print(f"  Val Loss: {history_finetune.history['val_loss'][epoch]:.4f}")
    print(f"  Val Accuracy: {history_finetune.history['val_accuracy'][epoch]:.4f}")
    print(f"  Val Precision: {history_finetune.history['val_precision'][epoch]:.4f}")
    print(f"  Val Recall: {history_finetune.history['val_recall'][epoch]:.4f}")
    print(f"  Val AUC: {history_finetune.history['val_auc'][epoch]:.4f}")

# 10. Save the final model after fine-tuning
model.save("D:/Deepfake_Detection_project/classification/models/model_500/deepfake_model_final_18.h5")
print("Final model saved!")

# 11. Load the best model for evaluation
best_model = keras.models.load_model("D:/Deepfake_Detection_project/classification/models/model_500/deepfake_model_best_18.h5")
print("Best model loaded for evaluation!")

# 12. Evaluate the best model on the test set
test_generator = data_generator(test_df, batch_size)
steps = len(test_df) // batch_size
loss, accuracy, precision, recall, auc = best_model.evaluate(test_generator, steps=steps)
print(f"Test set accuracy: {accuracy * 100:.2f}%")
print(f"Test set precision: {precision * 100:.2f}%")
print(f"Test set recall: {recall * 100:.2f}%")
print(f"Test set AUC: {auc * 100:.2f}%")

# Calculate the F1-score manually
y_true = []
y_pred = []
for _ in range(steps):
    images, labels = next(test_generator)
    preds = best_model.predict(images)
    y_true.extend(labels.flatten())
    y_pred.extend((preds > 0.5).astype(int).flatten())

f1 = f1_score(y_true, y_pred)
print(f"Test set F1-score: {f1 * 100:.2f}%")

# 13. Plot the Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.title("Confusion Matrix on Test Set (Best Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("D:/Deepfake_Detection_project/classification/output/confusion_matrix3_best.png")
plt.show()

# 14. Plot the training history (combine pre-training and fine-tuning)
# Kết hợp lịch sử huấn luyện từ pre-training và fine-tuning
history_combined = {}
for key in history_pretrain.history:
    history_combined[key] = history_pretrain.history[key] + history_finetune.history[key]

plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history_combined['loss'], label='Training Loss')
plt.plot(history_combined['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs (Pre-Training + Fine-Tuning)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history_combined['accuracy'], label='Training Accuracy')
plt.plot(history_combined['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs (Pre-Training + Fine-Tuning)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig("D:/Deepfake_Detection_project/classification/output/training_plots3_combined.png")
plt.show()