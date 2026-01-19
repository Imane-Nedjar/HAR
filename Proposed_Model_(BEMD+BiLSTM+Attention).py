#Display correlation matrix -Final
# =========================================================
# BEMD + BiLSTM + Attention (Real BIMFs Only = 5) 14/01/2025
# Full Corrected Version
# =========================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import CubicSpline

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score, log_loss
)

from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

# =========================================================
# 1. DATA LOADING FUNCTIONS
# =========================================================

def load_X(X_signals_paths):
    X_signals = []
    for signal_type_path in X_signals_paths:
        with open(signal_type_path, 'r') as file:
            X_signals.append(
                np.array([serie.split() for serie in file], dtype=np.float32)
            )
    return np.transpose(np.array(X_signals), (1, 2, 0))

def load_y(y_path):
    with open(y_path, 'r') as file:
        y_ = np.array([row.split() for row in file], dtype=np.int32)
    return y_ - 1

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
# 2. BEMD FUNCTION
# =========================================================

def bemd(x, y, max_imfs=5, n_directions=16, max_sift=5):
    t = np.arange(len(x))
    z = x + 1j * y
    residue = z.copy()
    bimfs = []

    for _ in range(max_imfs):
        h = residue.copy()
        for _ in range(max_sift):
            envelopes = []
            for theta in np.linspace(0, 2 * np.pi, n_directions, endpoint=False):
                proj = np.real(h * np.exp(-1j * theta))
                maxima = np.where((proj[1:-1] > proj[:-2]) & (proj[1:-1] > proj[2:]))[0] + 1
                minima = np.where((proj[1:-1] < proj[:-2]) & (proj[1:-1] < proj[2:]))[0] + 1
                extrema = np.sort(np.concatenate([maxima, minima]))

                if len(extrema) < 3:
                    continue

                cs = CubicSpline(t[extrema], proj[extrema])
                envelopes.append(cs(t))

            if len(envelopes) < 2:
                break
            h = h - np.mean(envelopes, axis=0)

        if np.std(np.real(h)) > 1e-7:
            bimfs.append(h)
            residue = residue - h
        else:
            break
    return bimfs

# =========================================================
# 3. FEATURE EXTRACTION
# =========================================================

def extract_bemd_features(X_data, n_imfs=5):
    processed = []
    print(f"Extracting {n_imfs} BIMFs per sample...")

    for sample in tqdm(X_data):
        # Apply BEMD to Body Acc (X,Y) and Body Gyro (X,Y)
        acc_bimfs = bemd(sample[:, 0], sample[:, 1], max_imfs=n_imfs)
        gyro_bimfs = bemd(sample[:, 3], sample[:, 4], max_imfs=n_imfs)

        def get_fixed_feats(bimf_list, target_n, original_signal):
            feats = [np.real(b) for b in bimf_list[:target_n]]
            while len(feats) < target_n:
                # Placeholder for missing IMFs using low-scale original signal
                feats.append(original_signal * 1e-4)
            return np.column_stack(feats)

        acc_f = get_fixed_feats(acc_bimfs, n_imfs, sample[:, 0])
        gyro_f = get_fixed_feats(gyro_bimfs, n_imfs, sample[:, 3])

        # Combined features: 9 (original) + 5 (acc IMFs) + 5 (gyro IMFs) = 19
        combined = np.column_stack((sample, acc_f, gyro_f))
        processed.append(combined)

    return np.array(processed)

# =========================================================
# 4. ATTENTION LAYER (FIXED BIAS SHAPE)
# =========================================================

class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is (batch_size, time_steps, feature_dim)
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1), # Shape must match time_steps
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

# =========================================================
# 5. DATA LOADING & PROCESSING
# =========================================================

X_train_raw = load_X([DATASET_PATH + "train/Inertial Signals/" + s + "train.txt" for s in INPUT_SIGNAL_TYPES])
X_test_raw = load_X([DATASET_PATH + "test/Inertial Signals/" + s + "test.txt" for s in INPUT_SIGNAL_TYPES])
y_train_raw = load_y(DATASET_PATH + "train/y_train.txt").flatten()
y_test_raw = load_y(DATASET_PATH + "test/y_test.txt").flatten()

# Feature extraction (This part is slow due to Cubic Splines)
X_train_bemd = extract_bemd_features(X_train_raw, n_imfs=5)
X_test_bemd = extract_bemd_features(X_test_raw, n_imfs=5)

# --- SAVE CORRELATION MATRIX ---
plt.figure(figsize=(12, 10))
df_corr = pd.DataFrame(X_train_bemd[0])
df_corr.columns = [i for i in range(1, df_corr.shape[1] + 1)]
sns.heatmap(df_corr.corr(), cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix (Indices 1 to 29)")
plt.savefig("Fifeature_correlation_bemd.png", dpi=300)
plt.show()


# Train/Val Split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_bemd, y_train_raw, test_size=0.2, random_state=42
)

y_train_cat = to_categorical(y_train, 6)
y_val_cat = to_categorical(y_val, 6)
y_test_cat = to_categorical(y_test_raw, 6)

# =========================================================
# 6. MODEL DEFINITION & TRAINING
# =========================================================

n_steps, n_input = X_train.shape[1], X_train.shape[2]

inputs = Input(shape=(n_steps, n_input))
x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
x = Dropout(0.5)(x)
x = Bidirectional(LSTM(32, return_sequences=True))(x)
x = Attention()(x)
x = Dense(64, activation="relu")(x)
outputs = Dense(6, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    X_train, y_train_cat, 
    epochs=30, batch_size=64, 
    validation_data=(X_val, y_val_cat)
)

# =========================================================
# 7. EVALUATION & METRICS
# =========================================================

y_probs = model.predict(X_test_bemd)
y_pred = np.argmax(y_probs, axis=1)

print("\n" + "=" * 30)
print("FINAL PERFORMANCE METRICS")
print("=" * 30)
print(classification_report(y_test_raw, y_pred, target_names=LABELS))

print(f"Accuracy      : {accuracy_score(y_test_raw, y_pred):.4f}")
print(f"Cohen's Kappa : {cohen_kappa_score(y_test_raw, y_pred):.4f}")
print(f"Log Loss      : {log_loss(y_test_cat, y_probs):.4f}")
print(f"F1-Score      : {f1_score(y_test_raw, y_pred, average='weighted'):.4f}")
print(f"Precision     : {precision_score(y_test_raw, y_pred, average='weighted'):.4f}")
print(f"Recall        : {recall_score(y_test_raw, y_pred, average='weighted'):.4f}")

# =========================================================
# 8. VISUALIZATION
# =========================================================

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_raw, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("Ficonfusion_matrix.png", dpi=300)
plt.show()

# Learning Curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Loss")
plt.legend()
plt.savefig("Filearning_curves.png", dpi=300)
plt.show()
