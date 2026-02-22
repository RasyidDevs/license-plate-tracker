import streamlit as st
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from models import YoloModel
from drawing import draw_results
from utils import uploaded_file_to_bgr, bgr_to_rgb

# ─── Page Config ───
st.set_page_config(
    page_title="🚗 Vehicle & License Plate Analyzer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Minimal custom CSS (theme handled by config.toml) ───
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.stButton > button {
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 0.65rem 2rem !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.5px;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(0, 224, 209, 0.3) !important;
}

div[data-testid="stFileUploader"] > div {
    border: 2px dashed rgba(0, 224, 209, 0.3);
    border-radius: 14px;
    transition: border-color 0.3s ease;
}
div[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(0, 224, 209, 0.7);
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Model paths ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAR_MODEL = os.path.join(BASE_DIR, "models", "car_detector.pt")
PLATE_SEG_MODEL = os.path.join(BASE_DIR, "models", "license_plate_segment.pt")


@st.cache_resource
def load_model():
    """Load and cache all YOLO models."""
    model = YoloModel(
        license_seg_model_path=PLATE_SEG_MODEL,
        car_model_path=CAR_MODEL,
    )
    model.load_model()
    return model


# ─── Sidebar ───
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    mode = st.radio(
        "🎯 Detection Mode",
        ["Detection", "Segmentation"],
        index=0,
        help="**Detection**: plate bounding box crop  \n**Segmentation**: segmented plate mask",
    )
    mode_val = mode.lower()

    st.divider()

    conf_car = st.slider(
        "🚗 Car Confidence", 0.1, 1.0, 0.5, 0.05,
        help="Minimum confidence for car detection",
    )
    conf_plate = st.slider(
        "🔖 Plate Confidence", 0.1, 1.0, 0.5, 0.05,
        help="Minimum confidence for plate detection",
    )

    st.divider()
    st.caption("🛠 YOLOv8 + Streamlit  •  © 2026")

# ─── Header ───
st.title("🚗 Vehicle & License Plate Analyzer")
st.caption("Upload gambar, lalu tekan **Analisis** untuk mendeteksi kendaraan dan plat nomor.")
st.divider()

# ─── File Upload (images only) ───
uploaded_files = st.file_uploader(
    "📁 Upload Images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="Supports JPG, PNG, WEBP",
)

# ─── Preview Grid ───
if uploaded_files:
    st.markdown("### 📷 Preview")
    cols_per_row = 3
    for i in range(0, len(uploaded_files), cols_per_row):
        batch = uploaded_files[i : i + cols_per_row]
        cols = st.columns(len(batch))
        for col, f in zip(cols, batch):
            with col:
                img = uploaded_file_to_bgr(f)
                st.image(bgr_to_rgb(img), caption=f"🖼 {f.name}", use_container_width=True)
    st.divider()

# ─── Analisis Button ───
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    analyze = st.button(
        "🔍  Analisis",
        use_container_width=True,
        disabled=(not uploaded_files),
    )

# ─── Run Analysis ───
if analyze and uploaded_files:
    model = load_model()
    model.conf_plate = conf_plate
    model.conf_car = conf_car

    st.divider()
    st.markdown("## 📊 Hasil Analisis")

    cols_per_row = min(len(uploaded_files), 3)

    for i in range(0, len(uploaded_files), cols_per_row):
        batch = uploaded_files[i : i + cols_per_row]
        cols = st.columns(len(batch))

        for col, f in zip(cols, batch):
            with col:
                with st.spinner(f"Analyzing {f.name}..."):
                    img_bgr = uploaded_file_to_bgr(f)
                    dets = model.predict_image(img_bgr, mode=mode_val)
                    annotated = draw_results(img_bgr, dets, mode=mode_val)

                    total_cars = sum(1 for d in dets if d["car_box"] is not None)
                    total_plates = sum(len(d["plates"]) for d in dets)

                st.image(bgr_to_rgb(annotated), caption=f.name, use_container_width=True)
                mc1, mc2 = st.columns(2)
                mc1.metric("🚗 Cars", total_cars)
                mc2.metric("🔖 Plates", total_plates)

elif not uploaded_files:
    st.markdown(
        """
        <div style="text-align:center; padding:3rem; opacity:0.5;">
            <div style="font-size:4rem;">📤</div>
            <h3>Upload gambar untuk memulai</h3>
            <p>JPG, PNG, WEBP</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
