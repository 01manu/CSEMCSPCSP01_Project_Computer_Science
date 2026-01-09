import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "best.keras")
JSON_PATH = os.path.join(BASE_DIR, "data", "label_num_to_disease_map.json")

IMG_SIZE = (224, 224)

with open(JSON_PATH) as f:
    label_map = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_image(image_path: str):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label = label_map[str(pred_idx)]

    print("Prediction:", label)
    print("Probabilities:")
    for i, p in enumerate(probs):
        print(f"  {label_map[str(i)]}: {p:.3f}")

if __name__ == "__main__":
    # Example usage: change path to one of your images
    test_img = os.path.join(BASE_DIR, "data", "train_images", "1000015157.jpg")
    predict_image(test_img)
