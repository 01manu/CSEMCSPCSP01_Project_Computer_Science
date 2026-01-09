import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from dataset import make_dataset
from model import build_model

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "train_images")
CSV_PATH = os.path.join(BASE_DIR, "data", "train.csv")
JSON_PATH = os.path.join(BASE_DIR, "data", "label_num_to_disease_map.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(CSV_PATH)

# -------------------------------
# TRAIN/VAL SPLIT
# -------------------------------
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(df["image_id"], df["label"]))

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df   = df.iloc[val_idx].reset_index(drop=True)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))

# -------------------------------
# CLASS WEIGHTS (to handle imbalance)
# -------------------------------
classes = np.sort(df["label"].unique())
weights = compute_class_weight(class_weight="balanced",
                               classes=classes,
                               y=train_df["label"].values)

class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
print("Class weights:", class_weights)

# -------------------------------
# DATASET BUILD
# -------------------------------
train_ds = make_dataset(train_df, DATA_DIR, augment=True, shuffle=True)
val_ds   = make_dataset(val_df, DATA_DIR, augment=False, shuffle=False)

# -------------------------------
# MODEL
# -------------------------------
model, base_model = build_model()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, "best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=3, monitor="val_accuracy",
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2
    )
]

# -------------------------------
# TRAINING (WARM-UP)
# -------------------------------
print("Starting warm-up training...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weights,
    callbacks=callbacks
)

# -------------------------------
# TRAINING (FINE-TUNING)
# -------------------------------
print("Fine-tuning base model...")
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weights,
    callbacks=callbacks
)

# -------------------------------
# SAVE FINAL MODEL
# -------------------------------
model.save(os.path.join(OUTPUT_DIR, "final.keras"))
print("Training completed.")
