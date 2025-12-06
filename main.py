# main.py
# Step 4: Load, preprocess, build and TRAIN a simple CNN for CIFAR-10
# Author: YOUR NAME

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

print("Loading CIFAR-10 dataset...")

# ----- 1. LOAD DATA -----
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Original training data shape:", x_train.shape, y_train.shape)
print("Original test data shape:", x_test.shape, y_test.shape)

# ----- 2. PREPROCESSING -----
# Normalise pixel values to 0–1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print("After preprocessing:")
print("x_train shape:", x_train.shape)
print("y_train_cat shape:", y_train_cat.shape)
print("x_test shape:", x_test.shape)
print("y_test_cat shape:", y_test_cat.shape)

# ----- 3. BUILD CNN MODEL -----
model = models.Sequential([
    # First convolution + pooling
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # Second convolution + pooling
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Flatten + fully connected layers
    layers.Flatten(),
    layers.Dense(64, activation="relu"),

    # Output layer: 10 classes with softmax
    layers.Dense(num_classes, activation="softmax")
])

# Compile model (tell it how to learn)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel summary:")
model.summary()

# ----- 4. TRAIN THE MODEL -----
print("\nTraining model...")
history = model.fit(
    x_train,
    y_train_cat,
    epochs=10,          # you can change this later if training is slow
    batch_size=64,
    validation_split=0.2
)

print("\nTraining finished.")