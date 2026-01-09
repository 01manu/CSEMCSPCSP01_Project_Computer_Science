import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # project root
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "best.keras")
LABELS_PATH = os.path.join(BASE_DIR, "data", "label_num_to_disease_map.json")

IMG_SIZE = (224, 224)

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Cassava Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Cassava Leaf Disease Detection")
st.write("Upload a cassava leaf image to predict the disease class.")

# -------------------------
# Load label map
# -------------------------
@st.cache_data
def load_label_map(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)

label_map = load_label_map(LABELS_PATH)
model = load_model(MODEL_PATH)

# -------------------------
# Preprocess image
# -------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
    return x

# -------------------------
# UI: File uploader
# -------------------------
uploaded = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    x = preprocess_image(img)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = label_map[str(pred_idx)]
    confidence = float(probs[pred_idx])

    st.subheader("Prediction Result")

    st.success(f"Predicted Disease: **{pred_label}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    # -------------------------
    # Top-3 Predictions
    # -------------------------
    st.subheader("Top-3 Predictions")

    sorted_items = sorted(
        [(label_map[str(i)], float(probs[i])) for i in range(len(probs))],
        key=lambda t: t[1],
        reverse=True
    )

    for i, (name, p) in enumerate(sorted_items[:3], start=1):
        st.write(f"**{i}. {name}** — {p:.2%}")
        st.progress(min(max(p, 0.0), 1.0))

