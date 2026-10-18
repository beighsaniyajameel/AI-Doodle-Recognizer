
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models, callbacks

# 1. Paths
DATA_PATH = Path("dataset/processed/doodle_data.npz")
MODEL_PATH = Path("models/doodle_cnn_improved.keras")

# Reproducibility
tf.keras.utils.set_random_seed(42)

# 2. Load dataset
print("Loading dataset...")
data = np.load(DATA_PATH)

X_train = data["X_train"].astype("float32")
y_train = data["y_train"].astype("int32")
X_test = data["X_test"].astype("float32")
y_test = data["y_test"].astype("int32")

# 3. Normalize and reshape images
# QuickDraw pixel values are expected to be in the range 0–255.
if X_train.max() > 1.0:
    X_train /= 255.0
    X_test /= 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Image shape:", X_train.shape[1:])

# 4. Data augmentation
# Small changes help the model generalize to different drawings.
augmentation = models.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomZoom(0.08),
], name="data_augmentation")

# 5. Build an improved CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    augmentation,

    # Convolution block 1
    layers.Conv2D(32, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(32, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.15),

    # Convolution block 2
    layers.Conv2D(64, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(64, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.20),

    # Convolution block 3
    layers.Conv2D(128, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.GlobalAveragePooling2D(),

    # Classification head
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.35),
    layers.Dense(10, activation="softmax")
])

# 6. Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 7. Training callbacks
training_callbacks = [
    callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
]

# 8. Train
print("\nStarting improved CNN training...")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=training_callbacks,
    verbose=1
)

# 9. Evaluate on the untouched test set
print("\nEvaluating improved model...")
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=1
)

print(f"\nImproved CNN test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

# 10. Save separately to preserve the original model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)

print(f"\nImproved model saved to: {MODEL_PATH}")
print("Training complete!")
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Class names must match the label order used during preprocessing
CLASSES = [
    "cat", "dog", "car", "house", "tree",
    "apple", "fish", "bicycle", "flower", "star"
]

# Get predictions for the untouched test set
y_prob = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

# Print precision, recall, and F1-score for each class
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    zero_division=0
))

# Plot confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(CLASSES)))
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES
)
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.title("CNN Confusion Matrix")
plt.tight_layout()
plt.show()