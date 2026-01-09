import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(DATA_DIR, "train_images")
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "best.keras")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_STATE = 42


def load_label_map():
    # If you already have label_num_to_disease_map.json and use it elsewhere, you can load it too.
    # Here we keep a safe default mapping order used in your report:
    return {
        0: "Cassava Bacterial Blight (CBB)",
        1: "Cassava Brown Streak Disease (CBSD)",
        2: "Cassava Green Mottle (CGM)",
        3: "Cassava Mosaic Disease (CMD)",
        4: "Healthy",
    }


def preprocess_image(path: tf.Tensor):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img


def make_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (preprocess_image(p), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    label_map = load_label_map()

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    df["path"] = df["image_id"].apply(lambda x: os.path.join(IMG_DIR, x))

    # Stratified split (same method as training)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(splitter.split(df["path"], df["label"]))
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Predict on validation set
    val_paths = val_df["path"].values
    y_true = val_df["label"].values.astype(int)

    ds = make_dataset(val_paths, y_true)
    probs = model.predict(ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    # Confusion matrix (for analysis here)
    cm = confusion_matrix(y_true, y_pred, labels=list(label_map.keys()))

    # Build misclassified table
    mis_mask = y_pred != y_true
    mis_df = val_df.loc[mis_mask, ["image_id", "path", "label"]].copy()
    mis_df["true_name"] = mis_df["label"].map(label_map)
    mis_df["pred"] = y_pred[mis_mask]
    mis_df["pred_name"] = mis_df["pred"].map(label_map)
    mis_df["pred_conf"] = probs[mis_mask, :].max(axis=1)

    # Save CSV of misclassified samples
    mis_csv_path = os.path.join(OUT_DIR, "misclassified_samples.csv")
    mis_df[["image_id", "true_name", "pred_name", "pred_conf"]].to_csv(mis_csv_path, index=False)

    # Error summary by class (helps minority class discussion)
    summary = []
    for k, name in label_map.items():
        total = int((y_true == k).sum())
        wrong = int(((y_true == k) & (y_pred != k)).sum())
        acc = 0.0 if total == 0 else (total - wrong) / total
        summary.append({"class": name, "val_samples": total, "misclassified": wrong, "class_accuracy": acc})

    summary_df = pd.DataFrame(summary).sort_values("class_accuracy")
    summary_path = os.path.join(OUT_DIR, "error_summary_by_class.csv")
    summary_df.to_csv(summary_path, index=False)

    # Create a figure: grid of misclassified examples (evidence for report)
    n_show = min(12, len(mis_df))
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    # Choose the most confident wrong predictions (interesting error cases)
    show_df = mis_df.sort_values("pred_conf", ascending=False).head(n_show).reset_index(drop=True)

    for i in range(12):
        ax = axes[i]
        ax.axis("off")
        if i >= n_show:
            continue

        row = show_df.iloc[i]
        img = tf.keras.utils.load_img(row["path"], target_size=IMG_SIZE)
        ax.imshow(img)
        title = f"T: {row['true_name']}\nP: {row['pred_name']} ({row['pred_conf']:.2f})"
        ax.set_title(title, fontsize=9)

    fig.suptitle("Misclassified Validation Examples (Top Confident Errors)", fontsize=14)
    fig.tight_layout()

    out_fig_path = os.path.join(OUT_DIR, "misclassified_examples.png")
    plt.savefig(out_fig_path, dpi=200)
    plt.close(fig)

    print("Saved:")
    print(" -", mis_csv_path)
    print(" -", summary_path)
    print(" -", out_fig_path)

    # OPTIONAL: print top confusion pairs (quick insight)
    print("\nTop confusions (true -> predicted):")
    for true_cls in label_map.keys():
        row = cm[true_cls].copy()
        row[true_cls] = 0
        pred_cls = int(np.argmax(row))
        if row[pred_cls] > 0:
            print(f" - {label_map[true_cls]} -> {label_map[pred_cls]} : {row[pred_cls]} cases")


if __name__ == "__main__":
    main()
