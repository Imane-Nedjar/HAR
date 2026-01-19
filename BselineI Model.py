# =========================================================
# BiLSTM on RAW SIGNALS (WITHOUT BEMD, WITHOUT ATTENTION)
# =========================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score, log_loss
)

from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# =========================================================
# 1. DATA LOADING FUNCTIONS
# =========================================================

def load_X(X_signals_paths):
    X_signals = []
    for path in X_signals_paths:
        with open(path, 'r') as file:
            X_signals.append(
                np.array([row.split() for row in file], dtype=np.float32)
            )
    return np.transpose(np.array(X_signals), (1, 2, 0))

def load_y(y_path):
    with open(y_path, 'r') as file:
        y_ = np.array([row.split() for row in file], dtype=np.int32)
    return y_.flatten() - 1

# =========================================================
# 2. DATASET CONFIGURATION
# =========================================================

DATASET_PATH = "/kaggle/input/human-activity-recognition/UCI_HAR_Dataset/"

INPUT_SIGNAL_TYPES = [
    "body_acc_x_", "body_acc_y_", "body_acc_z_",
    "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
    "total_acc_x_", "total_acc_y_", "total_acc_z_"
]

LABELS = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING"
]

# =========================================================
# 3. LOAD RAW DATA (NO BEMD)
# =========================================================

X_train = load_X([
    DATASET_PATH + "train/Inertial Signals/" + s + "train.txt"
    for s in INPUT_SIGNAL_TYPES
])

X_test = load_X([
    DATASET_PATH + "test/Inertial Signals/" + s + "test.txt"
    for s in INPUT_SIGNAL_TYPES
])

y_train = load_y(DATASET_PATH + "train/y_train.txt")
y_test = load_y(DATASET_PATH + "test/y_test.txt")

# =========================================================
# 4. CORRELATION MATRIX (RAW SIGNALS – ONE SAMPLE)
# =========================================================

plt.figure(figsize=(10, 8))
df_corr = pd.DataFrame(X_train[0])
sns.heatmap(df_corr.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Matrix – Raw Sensor Signals")
plt.savefig("Withoutfeature_correlation_raw.png", dpi=300)
plt.show()

# =========================================================
# 5. TRAIN / VALIDATION SPLIT
# =========================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

y_train_cat = to_categorical(y_train, 6)
y_val_cat = to_categorical(y_val, 6)
y_test_cat = to_categorical(y_test, 6)

# =========================================================
# 6. BiLSTM MODEL (RAW SIGNALS)
# =========================================================

n_steps, n_input = X_train.shape[1], X_train.shape[2]

inputs = Input(shape=(n_steps, n_input))

x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
x = Dropout(0.5)(x)
x = Bidirectional(LSTM(32, return_sequences=False))(x)

x = Dense(64, activation="relu")(x)
outputs = Dense(6, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train, y_train_cat,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val_cat)
)

# =========================================================
# 7. EVALUATION
# =========================================================

y_probs = model.predict(X_test)
y_pred = np.argmax(y_probs, axis=1)

print("\nFINAL PERFORMANCE METRICS\n")
print(classification_report(y_test, y_pred, target_names=LABELS))

print(f"Accuracy      : {accuracy_score(y_test, y_pred):.4f}")
print(f"Cohen Kappa   : {cohen_kappa_score(y_test, y_pred):.4f}")
print(f"Log Loss      : {log_loss(y_test_cat, y_probs):.4f}")
print(f"F1-score      : {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Precision     : {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall        : {recall_score(y_test, y_pred, average='weighted'):.4f}")

# =========================================================
# 8. VISUALIZATION
# =========================================================

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Raw Signals")
plt.savefig("Withoutconfusion_matrix_raw.png", dpi=300)
plt.show()

# Learning Curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.legend()

plt.savefig("Withoutlearning_curves_raw.png", dpi=300)
plt.show()