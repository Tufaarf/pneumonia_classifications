import os
import io
import glob
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Pneumonia Classifier", page_icon="🩺", layout="centered")

# --- Konfigurasi Utama ---
IMG_SIZE = (224, 224)

# --- PERBAIKAN FINAL ---
# Mengembalikan urutan label sesuai dengan urutan abjad folder
# yang dipelajari oleh Keras saat training.
# 'NORMAL' -> 0, 'PNEUMONIA BACTERI' -> 1, 'PNEUMONIA VIRUS' -> 2
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA BACTERI", 2: "PNEUMONIA VIRUS"}

DEFAULT_MODELS_DIR = "models"

# --- Fungsi-Fungsi Helper ---

@st.cache_resource(show_spinner="Memuat model Keras...")
def load_keras_model(model_path: str):
    """Memuat model Keras dari path file."""
    try:
        # Muat model tanpa kompilasi untuk mempercepat inferensi
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Melakukan preprocessing pada gambar PIL untuk input model."""
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # Menambah dimensi batch
    return arr

def list_models(dir_path: str) -> list:
    """Mencari file model .h5 atau .keras di direktori yang diberikan."""
    if not os.path.isdir(dir_path):
        return []
    patterns = [
        os.path.join(dir_path, "*.h5"),
        os.path.join(dir_path, "*.keras"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(list(set(files)))

# --- UI Sidebar ---
st.sidebar.header("🔧 Pengaturan Model")
available_models = list_models(DEFAULT_MODELS_DIR)

model_source = st.sidebar.radio(
    "Sumber model:",
    ["Pilih dari folder 'models/'", "Upload file model"],
    index=0 if available_models else 1,
    help="Pilih model yang sudah ada atau unggah file .h5 atau .keras Anda sendiri."
)

MODEL_OBJ = None
selected_model_path = None

if model_source == "Pilih dari folder 'models/'":
    if not available_models:
        st.sidebar.warning(f"Tidak ada file model di folder '{DEFAULT_MODELS_DIR}/'. Silakan gunakan opsi upload.")
    else:
        selected_model_path = st.sidebar.selectbox(
            "Pilih file model:",
            available_models,
            index=0,
            format_func=lambda p: os.path.basename(p),
        )
        if selected_model_path:
            MODEL_OBJ = load_keras_model(selected_model_path)
else:
    uploaded_file = st.sidebar.file_uploader("Upload file model (.h5 atau .keras)", type=["h5", "keras"])
    if uploaded_file is not None:
        temp_model_path = os.path.join(".", uploaded_file.name)
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        MODEL_OBJ = load_keras_model(temp_model_path)
        selected_model_path = temp_model_path

# --- UI Utama ---
st.title("🩺 Klasifikasi Pneumonia dari Citra X-ray")
st.caption("Aplikasi ini mengklasifikasikan citra X-ray dada ke dalam 3 kategori: Normal, Pneumonia Bakteri, atau Pneumonia Virus.")

if MODEL_OBJ is None:
    st.info("Silakan pilih atau upload model yang valid di sidebar untuk memulai.")
    st.stop()

st.subheader("🖼️ Prediksi Gambar")
mode = st.radio("Mode input:", ["Satu gambar", "Beberapa gambar"], horizontal=True)

if mode == "Satu gambar":
    img_file = st.file_uploader("Upload 1 gambar X-ray (jpg/png)", type=["jpg", "jpeg", "png"])
    if img_file:
        pil_image = Image.open(img_file)
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(pil_image, caption="Gambar Input", use_container_width=True)

        with col2:
            with st.spinner("Mengklasifikasikan..."):
                processed_img = preprocess_image(pil_image)
                probabilities = MODEL_OBJ.predict(processed_img, verbose=0)[0]
                
                pred_idx = np.argmax(probabilities)
                pred_label = CLASS_NAMES[pred_idx]
                confidence = probabilities[pred_idx]

                st.metric(
                    label="Hasil Prediksi",
                    value=pred_label,
                    help=f"Model memprediksi gambar ini sebagai '{pred_label}' dengan tingkat kepercayaan {confidence:.2%}.",
                )
                
                st.write("**Tingkat Kepercayaan:**")
                st.progress(float(confidence), text=f"{confidence:.2%}")
                
                st.write("**Probabilitas per Kelas:**")
                prob_df = pd.DataFrame({
                    'Kelas': [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))],
                    'Probabilitas': probabilities
                })
                st.bar_chart(prob_df.set_index('Kelas'))
else: 
    img_files = st.file_uploader("Upload beberapa gambar (jpg/png)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    if img_files:
        results = []
        progress_bar = st.progress(0, text="Memproses gambar...")
        for i, f in enumerate(img_files):
            pil_image = Image.open(f).convert("RGB")
            processed_img = preprocess_image(pil_image)
            
            probabilities = MODEL_OBJ.predict(processed_img, verbose=0)[0]
            pred_idx = np.argmax(probabilities)
            label = CLASS_NAMES[pred_idx]
            confidence = probabilities[pred_idx]

            results.append((f.name, label, confidence, pil_image))
            progress_bar.progress((i + 1) / len(img_files), text=f"Memproses {f.name}...")
        
        progress_bar.empty()
        st.success(f"Selesai memproses {len(img_files)} gambar!")

        st.write("---")
        st.write("### Hasil Prediksi Batch")
        for name, label, conf, img in results:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(img, width=100)
            with col2:
                st.write(f"**File:** `{name}`")
                st.write(f"**Prediksi:** **{label}** (Kepercayaan: `{conf:.2%}`)")
            st.write("---")

with st.expander("ℹ️ Tampilkan Info Model"):
    if selected_model_path:
        st.write(f"**Path model:** `{os.path.basename(selected_model_path)}`")
    else:
        st.write("**Model di-upload dari file uploader.**")
    
    summary_lines = []
    try:
        MODEL_OBJ.summary(print_fn=lambda x: summary_lines.append(x))
        st.text("\n".join(summary_lines))
    except Exception:
        st.write("Tidak dapat menampilkan ringkasan model.")

st.caption("Catatan: Preprocessing gambar mencakup resize ke 224x224, konversi ke RGB, dan penskalaan piksel ke rentang [0, 1].")

