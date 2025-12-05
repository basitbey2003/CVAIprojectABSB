# main.py
# Step 1: Load CIFAR-10 dataset
# Author: YOUR NAME

import tensorflow as tf

print("Loading CIFAR-10 dataset...")

# CIFAR-10: 60,000 colour images (32x32) in 10 classes
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Training data shape:", x_train.shape, y_train.shape)
print("Test data shape:", x_test.shape, y_test.shape)
