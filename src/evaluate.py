import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "train_images")
CSV_PATH = os.path.join(BASE_DIR, "data", "train.csv")
JSON_PATH = os.path.join(BASE_DIR, "data", "label_num_to_disease_map.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# -------------------------------
# LOAD DATA + LABEL MAP
# -------------------------------
df = pd.read_csv(CSV_PATH)

with open(JSON_PATH) as f:
    label_map = json.load(f)  # e.g. {"0": "Cassava Bacterial Blight", ...}

# -------------------------------
# RECREATE VALIDATION SPLIT
# (same as in train.py)
# -------------------------------
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, val_idx = next(sss.split(df["image_id"], df["label"]))
val_df = df.iloc[val_idx].reset_index(drop=True)

print("Validation samples:", len(val_df))

# -------------------------------
# LOAD IMAGES INTO MEMORY
# -------------------------------
IMG_SIZE = (224, 224)

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

image_paths = val_df["image_id"].apply(lambda x: os.path.join(DATA_DIR, x)).tolist()
X = tf.stack([load_img(p) for p in image_paths])

y_true = val_df["label"].values

# -------------------------------
# LOAD MODEL
# -------------------------------
model_path = os.path.join(OUTPUT_DIR, "best.keras")
print("Loading model from:", model_path)
model = tf.keras.models.load_model(model_path, compile=False)

# -------------------------------
# PREDICTIONS
# -------------------------------
probs = model.predict(X, batch_size=32, verbose=1)
y_pred = np.argmax(probs, axis=1)

# -------------------------------
# CLASSIFICATION REPORT
# -------------------------------
target_names = [label_map[str(i)] for i in sorted(df["label"].unique())]
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
cm = confusion_matrix(y_true, y_pred, labels=sorted(df["label"].unique()))

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Cassava Leaf Disease - Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=200)
print("Confusion matrix saved to:", cm_path)
plt.close()
