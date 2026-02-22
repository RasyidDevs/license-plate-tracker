import streamlit as st
import cv2
import numpy as np
import os
import sys
import math

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from models import YoloModel
from drawing import draw_results
from utils import (
    is_image_file,
    is_video_file,
    uploaded_file_to_bgr,
    get_video_first_frame,
    get_video_info,
    video_frame_generator,
    uploaded_file_to_temp_path,
    bgr_to_rgb,
)

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
PLATE_DET_MODEL = os.path.join(BASE_DIR, "models", "license_plate_detector.pt")
PLATE_SEG_MODEL = os.path.join(BASE_DIR, "models", "license_plate_segment.pt")


@st.cache_resource
def load_model():
    """Load and cache all YOLO models."""
    model = YoloModel(
        license_det_model_path=PLATE_DET_MODEL,
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
st.caption("Upload images or videos, then press **Analisis** to detect vehicles and license plates.")
st.divider()

# ─── File Upload (multi-file) ───
uploaded_files = st.file_uploader(
    "📁 Upload Images or Videos",
    type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "mkv"],
    accept_multiple_files=True,
    help="Supports JPG, PNG, WEBP (images) — MP4, AVI, MOV, MKV (videos)",
)

# ─── Preview Grid ───
if uploaded_files:
    images_files = [f for f in uploaded_files if is_image_file(f.name)]
    video_files = [f for f in uploaded_files if is_video_file(f.name)]

    # Show preview in grid (up to 3 per row)
    if images_files or video_files:
        st.markdown("### 📷 Preview")
        all_previews = []

        for f in images_files:
            img = uploaded_file_to_bgr(f)
            all_previews.append((f.name, bgr_to_rgb(img), "image"))

        for f in video_files:
            frame = get_video_first_frame(f)
            if frame is not None:
                info = get_video_info(f)
                caption = f"{f.name} (frame 1 of {info['total_frames']})"
                all_previews.append((caption, bgr_to_rgb(frame), "video"))

        cols_per_row = 3
        for i in range(0, len(all_previews), cols_per_row):
            batch = all_previews[i : i + cols_per_row]
            cols = st.columns(len(batch))
            for col, (name, preview_rgb, ftype) in zip(cols, batch):
                with col:
                    icon = "🎥" if ftype == "video" else "🖼"
                    st.image(preview_rgb, caption=f"{icon} {name}", use_container_width=True)

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

    images_files = [f for f in uploaded_files if is_image_file(f.name)]
    video_files = [f for f in uploaded_files if is_video_file(f.name)]

    st.divider()
    st.markdown("## 📊 Hasil Analisis")

    # ─── Process Images ───
    if images_files:
        st.markdown("### 🖼 Images")

        cols_per_row = min(len(images_files), 3)

        for i in range(0, len(images_files), cols_per_row):
            batch = images_files[i : i + cols_per_row]
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

    # ─── Process Videos ───
    if video_files:
        st.markdown("### 🎥 Videos")

        for f in video_files:
            st.markdown(f"**{f.name}**")
            info = get_video_info(f)

            with st.spinner(f"Analyzing {f.name} ({info['total_frames']} frames)..."):
                frame_display = st.empty()
                progress = st.progress(0)
                total = max(info["total_frames"], 1)

                f.seek(0)
                frame_count = 0
                max_cars = 0
                max_plates = 0

                for frame_bgr in video_frame_generator(f):
                    dets = model.predict_frame(frame_bgr, mode=mode_val)
                    annotated = draw_results(frame_bgr, dets, mode=mode_val)

                    frame_display.image(
                        bgr_to_rgb(annotated),
                        caption=f"Frame {frame_count + 1} / {info['total_frames']}",
                        use_container_width=True,
                    )

                    nc = sum(1 for d in dets if d["car_box"] is not None)
                    np_ = sum(len(d["plates"]) for d in dets)
                    max_cars = max(max_cars, nc)
                    max_plates = max(max_plates, np_)
                    frame_count += 1
                    progress.progress(min(frame_count / total, 1.0))

                progress.empty()

            vc1, vc2, vc3, vc4 = st.columns(4)
            vc1.metric("🎞 Frames", frame_count)
            vc2.metric("⏱ FPS", f"{info['fps']:.0f}")
            vc3.metric("🚗 Max Cars/Frame", max_cars)
            vc4.metric("🔖 Max Plates/Frame", max_plates)
            st.divider()

elif not uploaded_files:
    st.markdown(
        """
        <div style="text-align:center; padding:3rem; opacity:0.5;">
            <div style="font-size:4rem;">📤</div>
            <h3>Upload gambar atau video untuk memulai</h3>
            <p>JPG, PNG, WEBP, MP4, AVI, MOV, MKV</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
