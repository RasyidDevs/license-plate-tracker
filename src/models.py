from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any, Tuple
import cv2
import tempfile


class YoloModel:
    def __init__(
        self,
        license_det_model_path: str,
        license_seg_model_path: str,
        car_model_path: str,
        conf_plate: float = 0.5,
        conf_car: float = 0.5,
    ):
        self.license_det_model_path = license_det_model_path
        self.license_seg_model_path = license_seg_model_path
        self.car_model_path = car_model_path
        self.conf_plate = conf_plate
        self.conf_car = conf_car
        self.car_model = None
        self.plate_det_model = None
        self.plate_seg_model = None

    def load_model(self):
        """Load all YOLO models and store on self."""
        self.car_model = YOLO(self.car_model_path)
        self.plate_det_model = YOLO(self.license_det_model_path)
        self.plate_seg_model = YOLO(self.license_seg_model_path)

    @staticmethod
    def _clamp_box(x1, y1, x2, y2, w, h):
        """Clamp bounding box coordinates to image dimensions."""
        x1 = int(max(0, min(x1, w - 1)))
        y1 = int(max(0, min(y1, h - 1)))
        x2 = int(max(0, min(x2, w - 1)))
        y2 = int(max(0, min(y2, h - 1)))
        return x1, y1, x2, y2

    def _detect_plates_in_roi(
        self, roi: np.ndarray, offset_x: int, offset_y: int,
        w: int, h: int, mode: str
    ) -> List[Dict[str, Any]]:
        """Run plate detection on a ROI and return plate dicts with absolute coords."""
        plate_model = (
            self.plate_seg_model if mode == "segmentation" else self.plate_det_model
        )
        plate_results = plate_model.predict(roi, conf=self.conf_plate, verbose=False)[0]
        plates = []

        if plate_results.boxes is None or len(plate_results.boxes) == 0:
            return plates

        p_boxes = plate_results.boxes.xyxy.cpu().numpy()
        p_confs = plate_results.boxes.conf.cpu().numpy()

        for i, ((px1, py1, px2, py2), pconf) in enumerate(zip(p_boxes, p_confs)):
            abs_px1 = int(offset_x + px1)
            abs_py1 = int(offset_y + py1)
            abs_px2 = int(offset_x + px2)
            abs_py2 = int(offset_y + py2)
            abs_px1, abs_py1, abs_px2, abs_py2 = self._clamp_box(
                abs_px1, abs_py1, abs_px2, abs_py2, w, h
            )

            # crop plate from ROI (relative coords)
            rpx1, rpy1, rpx2, rpy2 = int(px1), int(py1), int(px2), int(py2)
            plate_crop = roi[rpy1:rpy2, rpx1:rpx2].copy()

            # segmentation mask crop — per-pixel
            plate_seg_crop = None
            if (
                mode == "segmentation"
                and plate_results.masks is not None
                and i < len(plate_results.masks.data)
            ):
                mask_raw = plate_results.masks.data[i].cpu().numpy()  # float 0-1
                # threshold to binary then resize to ROI size
                mask_bin = (mask_raw > 0.5).astype(np.uint8)
                mask_resized = cv2.resize(
                    mask_bin, (roi.shape[1], roi.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                # crop mask to plate bounding box region (same coords as plate_crop)
                mask_crop = mask_resized[rpy1:rpy2, rpx1:rpx2]
                plate_region = roi[rpy1:rpy2, rpx1:rpx2].copy()
                if mask_crop.shape[:2] == plate_region.shape[:2] and mask_crop.any():
                    # Find tight bbox of mask pixels inside the plate bbox
                    ys, xs = np.where(mask_crop)
                    y1m, y2m = int(ys.min()), int(ys.max()) + 1
                    x1m, x2m = int(xs.min()), int(xs.max()) + 1

                    # Clamp to mask_crop bounds (safety)
                    y1m = max(0, y1m)
                    x1m = max(0, x1m)
                    y2m = min(mask_crop.shape[0], y2m)
                    x2m = min(mask_crop.shape[1], x2m)

                    # Extract tight region and apply mask so only plate pixels remain
                    mask_sub = mask_crop[y1m:y2m, x1m:x2m]
                    region_sub = plate_region[y1m:y2m, x1m:x2m]
                    if mask_sub.size > 0 and mask_sub.any():
                        plate_seg_crop = cv2.bitwise_and(
                            region_sub,
                            region_sub,
                            mask=(mask_sub * 255).astype(np.uint8),
                        )
                    else:
                        plate_seg_crop = None

            plates.append({
                "plate_box_abs": (abs_px1, abs_py1, abs_px2, abs_py2),
                "plate_conf": float(pconf),
                "plate_crop": plate_crop,
                "plate_seg_crop": plate_seg_crop,
            })

        return plates

    def predict_frame(
        self, frame_bgr: np.ndarray, mode: str = "detection"
    ) -> List[Dict[str, Any]]:
        """
        Detect cars and plates. If no cars found, run plate detection on full image.
        Returns list of dicts with car_box, car_conf, plates.
        """
        assert self.car_model is not None, "Call load_model() first"
        h, w = frame_bgr.shape[:2]

        # 1. Detect cars
        results = self.car_model.predict(frame_bgr, conf=self.conf_car, verbose=False)
        r = results[0]
        out = []

        has_cars = r.boxes is not None and len(r.boxes) > 0

        if has_cars:
            car_boxes = r.boxes.xyxy.cpu().numpy()
            car_confs = r.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cconf in zip(car_boxes, car_confs):
                x1, y1, x2, y2 = self._clamp_box(x1, y1, x2, y2, w, h)
                car_roi = frame_bgr[y1:y2, x1:x2]
                if car_roi.size == 0:
                    continue

                plates = self._detect_plates_in_roi(car_roi, x1, y1, w, h, mode)
                out.append({
                    "car_box": (x1, y1, x2, y2),
                    "car_conf": float(cconf),
                    "plates": plates,
                })
        else:
            # No cars detected → run plate detection on full image
            plates = self._detect_plates_in_roi(frame_bgr, 0, 0, w, h, mode)
            if plates:
                out.append({
                    "car_box": None,
                    "car_conf": 0.0,
                    "plates": plates,
                })

        return out

    def predict_image(
        self, img_bgr: np.ndarray, mode: str = "detection"
    ) -> List[Dict[str, Any]]:
        """Run prediction on a BGR numpy image."""
        return self.predict_frame(img_bgr, mode=mode)