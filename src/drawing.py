"""Drawing utilities for YOLO detection results overlay."""

import cv2
import numpy as np
from typing import List, Dict, Any


# Vibrant color palette
CAR_COLORS = [
    (0, 224, 209),    # cyan-teal
    (255, 107, 107),  # coral-red
    (78, 205, 196),   # mint
    (255, 195, 0),    # amber
    (162, 155, 254),  # lavender
    (0, 184, 148),    # green-teal
    (253, 121, 168),  # pink
    (108, 92, 231),   # purple
]

LINE_COLOR = (0, 255, 255)          # cyan for plate leader lines
PLATE_BORDER_COLOR = (0, 200, 255)  # orange-yellow for plate image border


def draw_results(
    frame_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    mode: str = "detection",
) -> np.ndarray:
    """
    Draw annotated results on frame:
    - Bounding box border on each vehicle (no fill)
    - For each plate: 2 leader lines from plate edges going upward,
      with the plate crop image displayed at the top of those lines.
    """
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]

    thickness = max(2, int(min(w, h) / 350))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(w, h) / 1400)
    txt_thick = max(1, int(font_scale * 2))

    for idx, det in enumerate(detections):
        color = CAR_COLORS[idx % len(CAR_COLORS)]

        # --- 1) Car bounding box (border only, no fill) ---
        if det["car_box"] is not None:
            cx1, cy1, cx2, cy2 = det["car_box"]
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), color, thickness)

            # Car label
            label = f"Car {det['car_conf']:.0%}"
            (tw, th_txt), _ = cv2.getTextSize(label, font, font_scale, txt_thick)
            lbl_y = cy1 - 4
            if lbl_y - th_txt < 0:
                lbl_y = cy1 + th_txt + 8
            cv2.rectangle(
                canvas,
                (cx1, lbl_y - th_txt - 6),
                (cx1 + tw + 8, lbl_y + 4),
                color, -1,
            )
            cv2.putText(
                canvas, label, (cx1 + 4, lbl_y),
                font, font_scale, (255, 255, 255), txt_thick, cv2.LINE_AA,
            )

        # --- 2) Draw plates with leader lines going upward ---
        for plate in det["plates"]:
            px1, py1, px2, py2 = plate["plate_box_abs"]

            # Choose plate image based on mode
            if mode == "segmentation" and plate.get("plate_seg_crop") is not None:
                plate_img = plate["plate_seg_crop"]
            else:
                plate_img = plate["plate_crop"]

            if plate_img is None or plate_img.size == 0:
                # Still draw plate bbox even if no image
                cv2.rectangle(canvas, (px1, py1), (px2, py2), LINE_COLOR, thickness)
                continue

            # Plate bounding box on the vehicle
            cv2.rectangle(canvas, (px1, py1), (px2, py2), LINE_COLOR, thickness)

            # --- Scale plate image for display ---
            plate_w = px2 - px1
            plate_display_w = max(plate_w, 80)
            plate_display_w = min(plate_display_w, int(w * 0.28))
            ph, pw_orig = plate_img.shape[:2]
            if pw_orig == 0 or ph == 0:
                continue
            scale = plate_display_w / pw_orig
            plate_display_h = max(int(ph * scale), 20)
            plate_display_h = min(plate_display_h, int(h * 0.18))
            plate_resized = cv2.resize(plate_img, (plate_display_w, plate_display_h))

            # --- Calculate line target (above the car / plate) ---
            ref_top = det["car_box"][1] if det["car_box"] is not None else py1
            line_gap = max(int(h * 0.03), 15)
            target_y_bottom = max(ref_top - line_gap, plate_display_h + 10)
            target_y_top = target_y_bottom - plate_display_h

            # Center the plate image horizontally above the plate box
            plate_center_x = (px1 + px2) // 2
            img_x1 = plate_center_x - plate_display_w // 2
            img_x2 = img_x1 + plate_display_w

            # Clamp horizontally
            if img_x1 < 0:
                img_x2 += -img_x1
                img_x1 = 0
            if img_x2 > w:
                img_x1 -= (img_x2 - w)
                img_x2 = w
            img_x1 = max(img_x1, 0)

            # Clamp vertically
            if target_y_top < 0:
                target_y_top = 0
                target_y_bottom = plate_display_h

            # --- Draw 2 leader lines ---
            # Left line:  plate left edge → image area left
            cv2.line(
                canvas, (px1, py1), (img_x1, target_y_bottom),
                LINE_COLOR, thickness, cv2.LINE_AA,
            )
            # Right line: plate right edge → image area right
            cv2.line(
                canvas, (px2, py1), (img_x2, target_y_bottom),
                LINE_COLOR, thickness, cv2.LINE_AA,
            )
            # Connecting horizontal line at top
            cv2.line(
                canvas, (img_x1, target_y_bottom), (img_x2, target_y_bottom),
                LINE_COLOR, thickness, cv2.LINE_AA,
            )
            if mode == "detection":
                # --- Draw border around plate image ---
                border = max(2, thickness)
                cv2.rectangle(
                    canvas,
                    (img_x1 - border, target_y_top - border),
                    (img_x2 + border, target_y_bottom + border),
                    PLATE_BORDER_COLOR, border,
                )

            # --- Paste the plate image ---
            try:
                roi_h = target_y_bottom - target_y_top
                roi_w = img_x2 - img_x1
                if roi_h > 0 and roi_w > 0:
                    paste = cv2.resize(plate_resized, (roi_w, roi_h))
                    canvas[target_y_top:target_y_bottom, img_x1:img_x2] = paste
            except Exception:
                pass

            # Plate confidence label
            plate_label = f"Plate {plate['plate_conf']:.0%}"
            (ptw, pth), _ = cv2.getTextSize(
                plate_label, font, font_scale * 0.8, txt_thick
            )
            label_y = target_y_top - 5
            if label_y - pth < 0:
                label_y = target_y_bottom + pth + 10
            cv2.putText(
                canvas, plate_label, (img_x1, label_y),
                font, font_scale * 0.8, LINE_COLOR, txt_thick, cv2.LINE_AA,
            )

    return canvas
