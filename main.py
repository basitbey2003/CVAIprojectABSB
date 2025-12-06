# main.py
# FINAL VERSION – CNN for CIFAR-10 (Ravensbourne Computer Vision & AI Coursework)
# Author: YOUR NAME

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

print("Loading CIFAR-10 dataset...")

# ----- 1. LOAD DATA -----
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Original training data shape:", x_train.shape, y_train.shape)
print("Original test data shape:", x_test.shape, y_test.shape)

# ----- 2. PREPROCESSING -----
# Normalise image pixel values to 0–1 for better training stability
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert label integers to one-hot encoded vectors
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print("After preprocessing:")
print("x_train shape:", x_train.shape)
print("y_train_cat shape:", y_train_cat.shape)
print("x_test shape:", x_test.shape)
print("y_test_cat shape:", y_test_cat.shape)

# Class names (for evaluation output)
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ----- 3. BUILD CNN MODEL -----
# A simple CNN with two convolutional blocks
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

# ----- 4. COMPILE MODEL -----
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel summary:")
model.summary()

# ----- 5. TRAIN MODEL -----
print("\nTraining model...")
history = model.fit(
    x_train,
    y_train_cat,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)
print("\nTraining complete.")

# ----- 6. EVALUATE MODEL -----
print("\nEvaluating on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Predictions for metrics
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test.flatten()

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ----- 7. SAVE TRAINED MODEL -----
save_dir = "saved_model"
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "cifar_cnn.h5")
model.save(model_path)

print(f"\nModel saved to: {model_path}")
print("\nALL TASKS COMPLETE ✔️")