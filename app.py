# app.py  — Streamlit-only version
import os
from io import BytesIO

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Pokémon Identifier", page_icon="🧪", layout="centered")
st.title("🧪 Pokémon Identifier")
st.write("Upload an image and I'll predict the Pokémon using your CNN model.")

# ---------------------------
# Paths / config
# ---------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "pokemon_classifier_dropout=8,6_lr=0.0001.h5")        # <- adjust if your model file lives elsewhere
LABELS_PATH = os.getenv("LABELS_PATH", "class_names.txt")  # One class name per line (optional)
IMAGE_SIZE = (224, 224)   # match what your model expects
TOPK = 3

# ---------------------------
# Utilities
# ---------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    # TF 2.12 uses tf.keras.*; keep it consistent
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_data
def load_class_names():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if names:
            return names
    # Fallback to generic indices if file not present
    return None

def preprocess(img_pil: Image.Image) -> np.ndarray:
    img = img_pil.convert("RGB").resize(IMAGE_SIZE)
    x = np.asarray(img, dtype=np.float32)
    # If you trained with MobileNet preprocessing:
    from tensorflow.keras.applications.mobilenet import preprocess_input
    x = preprocess_input(x)  # scales to MobileNet expected range
    x = np.expand_dims(x, axis=0)
    return x

def predict(img_pil: Image.Image):
    model = load_model()
    x = preprocess(img_pil)
    probs = model.predict(x, verbose=0)
    # Ensure shape [1, num_classes]
    if probs.ndim == 1:
        probs = np.expand_dims(probs, 0)
    return probs[0]  # vector of class probabilities

# ---------------------------
# UI
# ---------------------------
uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
if uploaded:
    # Show preview
    img = Image.open(BytesIO(uploaded.read()))
    st.image(img, caption="Uploaded image", use_column_width=True)
    st.divider()

    run = st.button("🔮 Predict")
    if run:
        with st.spinner("Running inference…"):
            try:
                probs = predict(img)
                class_names = load_class_names()
                num_classes = probs.shape[-1]
                topk = min(TOPK, num_classes)
                top_idx = np.argsort(probs)[::-1][:topk]
                top_probs = probs[top_idx]

                st.subheader("Results")
                for rank, (idx, p) in enumerate(zip(top_idx, top_probs), start=1):
                    name = class_names[idx] if class_names and idx < len(class_names) else f"class_{idx}"
                    st.write(f"**{rank}. {name}** — {p*100:.2f}%")

                # Nice little bar chart
                chart_data = { (class_names[i] if class_names and i < len(class_names) else f"class_{i}"): float(probs[i]) for i in top_idx }
                st.bar_chart(chart_data)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)
else:
    st.info("👆 Upload a Pokémon image to begin.")
