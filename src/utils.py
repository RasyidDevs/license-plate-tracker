"""Utility functions for file handling and image/video processing."""

import cv2
import numpy as np
import tempfile
from typing import List, Tuple, Optional


IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def get_file_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower()


def is_image_file(filename: str) -> bool:
    return get_file_extension(filename) in IMAGE_EXTENSIONS


def is_video_file(filename: str) -> bool:
    return get_file_extension(filename) in VIDEO_EXTENSIONS


def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    """Convert a Streamlit UploadedFile (image) to BGR numpy array."""
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def uploaded_file_to_temp_path(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a real temp file path."""
    suffix = "." + uploaded_file.name.split(".")[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    uploaded_file.seek(0)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    return tmp.name


def get_video_first_frame(uploaded_file) -> Optional[np.ndarray]:
    """Extract the first frame from a video UploadedFile as BGR numpy array."""
    tmp_path = uploaded_file_to_temp_path(uploaded_file)
    cap = cv2.VideoCapture(tmp_path)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def video_frame_generator(uploaded_file):
    """Generator yielding (frame_bgr) for each frame of a video."""
    tmp_path = uploaded_file_to_temp_path(uploaded_file)
    cap = cv2.VideoCapture(tmp_path)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def get_video_info(uploaded_file) -> dict:
    """Get total frames and FPS of a video."""
    tmp_path = uploaded_file_to_temp_path(uploaded_file)
    cap = cv2.VideoCapture(tmp_path)
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB for display in Streamlit."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
