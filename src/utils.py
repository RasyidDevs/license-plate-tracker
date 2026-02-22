"""Utility functions for file handling and image processing."""

import cv2
import numpy as np
from typing import List, Optional


def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    """Convert a Streamlit UploadedFile (image) to BGR numpy array."""
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB for display in Streamlit."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
