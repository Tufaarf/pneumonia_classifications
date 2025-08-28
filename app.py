# app.py
import os
import io
import glob
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Pneumonia Classifier", page_icon="🩺", layout="centered")

IMG_SIZE = (224, 224)
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"} 
DEFAULT_MODELS_DIR = "models"  

@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str):
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(pil_img: Image.Image) -> np.ndarray:

    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_single(model, pil_img: Image.Image) -> dict:
    x = preprocess_image(pil_img)

    prob_pneumonia = float(model.predict(x, verbose=0)[0][0])
    prob_normal = 1.0 - prob_pneumonia
    pred_idx = 1 if prob_pneumonia >= 0.5 else 0
    return {
        "pred_label": CLASS_NAMES[pred_idx],
        "prob_pneumonia": prob_pneumonia,
        "prob_normal": prob_normal
    }

def list_models(dir_path: str):
  
    if not os.path.isdir(dir_path):
        return []
    patterns = [
        os.path.join(dir_path, "Model_Skenario_*__*__R_*_TRAIN_RATIO_.h5"),
        os.path.join(dir_path, "Model_Skenario*_*.h5"),  
        os.path.join(dir_path, "*.h5"),             
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
 
    files = sorted(list(set(files)))
    return files


st.sidebar.header("🔧 Model")
available = list_models(DEFAULT_MODELS_DIR)

model_source = st.sidebar.radio(
    "Sumber model (.h5):",
    ["Pilih dari folder 'models/'", "Upload file .h5"],
    index=0 if available else 1,
)

model_bytes = None
selected_model_path = None

if model_source == "Pilih dari folder 'models/'":
    if not available:
        st.sidebar.warning("Tidak ada file .h5 di folder 'models/'. Gunakan opsi upload.")
    else:
        selected_model_path = st.sidebar.selectbox(
            "Pilih file model:",
            available,
            index=0,
            format_func=lambda p: os.path.basename(p),
        )
else:
    up = st.sidebar.file_uploader("Upload file model (.h5)", type=["h5"])
    if up is not None:
        model_bytes = up.read()


threshold = st.sidebar.slider("Ambang (threshold) PNEUMONIA", 0.1, 0.9, 0.5, 0.05)

st.title("🩺 Pneumonia X-ray Classifier")
st.caption("Memuat model Keras (.h5) dengan pola nama **Model_Skenario _NOMOR___OPTIMIZER__R_TRAIN_RATIO_.h5**")


MODEL_OBJ = None

if selected_model_path:
    with st.spinner(f"Memuat model: {os.path.basename(selected_model_path)}"):
        MODEL_OBJ = load_keras_model(selected_model_path)
elif model_bytes:
  
    tmp_path = "uploaded_model.h5"
    with open(tmp_path, "wb") as f:
        f.write(model_bytes)
    with st.spinner("Memuat model ter-upload..."):
        MODEL_OBJ = load_keras_model(tmp_path)

if MODEL_OBJ is None:
    st.info("Pilih atau upload model terlebih dahulu di sidebar.")
    st.stop()


st.subheader("🖼️ Prediksi Gambar")
mode = st.radio("Mode input:", ["Satu gambar", "Beberapa gambar"], horizontal=True)

if mode == "Satu gambar":
    img_file = st.file_uploader("Upload 1 gambar X-ray (jpg/png)", type=["jpg", "jpeg", "png"])
    if img_file:
        pil = Image.open(img_file)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(pil, caption="Input", use_container_width=True)
        with col2:
            with st.spinner("Mengklasifikasikan..."):
                x = preprocess_image(pil)
                prob_pneu = float(MODEL_OBJ.predict(x, verbose=0)[0][0])
                pred_idx = 1 if prob_pneu >= threshold else 0
                label = CLASS_NAMES[pred_idx]
                st.metric(
                    "Prediksi",
                    value=label,
                    delta=f"P(PNEUMONIA) = {prob_pneu:.3f}",
                    help=f"Threshold: {threshold:.2f} → {'PNEUMONIA' if prob_pneu>=threshold else 'NORMAL'}",
                )
                st.progress(min(max(prob_pneu, 0.0), 1.0), text="Probabilitas PNEUMONIA")
                st.write(
                    f"**P(NORMAL)**: `{1.0 - prob_pneu:.3f}`  •  **P(PNEUMONIA)**: `{prob_pneu:.3f}`"
                )
else:
    img_files = st.file_uploader("Upload beberapa gambar (jpg/png)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    if img_files:
        results = []
        for f in img_files:
            pil = Image.open(f).convert("RGB")
            x = preprocess_image(pil)
            prob_pneu = float(MODEL_OBJ.predict(x, verbose=0)[0][0])
            pred_idx = 1 if prob_pneu >= threshold else 0
            label = CLASS_NAMES[pred_idx]
            results.append((f.name, label, prob_pneu, 1.0 - prob_pneu))
   
        st.write("### Hasil")
        for name, label, p_pneu, p_norm in results:
            st.write(f"- **{name}** → **{label}**  |  P(PNEUMONIA) = `{p_pneu:.3f}`  •  P(NORMAL) = `{p_norm:.3f}`")

with st.expander("ℹ️ Info Model"):
    if selected_model_path:
        st.write(f"**Path model:** `{selected_model_path}`")
    else:
        st.write("**Model di-upload dari file uploader.**")
    try:
        MODEL_OBJ.summary(print_fn=lambda x: st.text(x))
    except Exception:
        st.write("Ringkasan model tidak tersedia (beberapa model terserialisasi tanpa arsitektur lengkap).")

st.caption("Catatan: Preprocessing mengikuti training (resize 224×224, skala 1/255, sigmoid). Label diasumsikan {0: NORMAL, 1: PNEUMONIA}.")
