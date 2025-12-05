# main.py
# Step 2: Load and preprocess CIFAR-10 dataset
# Author: YOUR NAME

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

print("Loading CIFAR-10 dataset...")

# CIFAR-10: 60,000 colour images (32x32) in 10 classes
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Original training data shape:", x_train.shape, y_train.shape)
print("Original test data shape:", x_test.shape, y_test.shape)

# ---------- PREPROCESSING ----------

# 1. Normalise images: make pixel values between 0 and 1 instead of 0–255
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 2. One-hot encode labels: turn numbers (0–9) into vectors of length 10
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print("After preprocessing:")
print("x_train shape:", x_train.shape)
print("y_train_cat shape:", y_train_cat.shape)
print("x_test shape:", x_test.shape)
print("y_test_cat shape:", y_test_cat.shape)