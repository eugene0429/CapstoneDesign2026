"""
Real-time YOLO inference on the RealSense D435i color stream.

Loads a trained YOLO weights file (default: latest
`perception/training/runs/*/weights/best.pt`), pulls aligned color+depth from
the camera, draws each detection's bbox with confidence and the depth (m) at
the bbox-center pixel, and overlays current FPS.

Keys (cv2 window in focus):
    q, Esc        quit

Usage:
    python -m perception.detection.realtime_infer
    python -m perception.detection.realtime_infer --conf 0.5
    python -m perception.detection.realtime_infer --weights path/to/best.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from perception.common.realsense_wrapper import RealSenseCamera
from perception.config import CAMERA

HERE = Path(__file__).resolve().parent
TRAINING_RUNS = HERE.parent / "training" / "runs"

IMGSZ = 640
DEVICE = "0"
FPS_EMA_ALPHA = 0.1


def find_latest_best(runs_root: Path) -> Path:
    """Return the newest `runs/*/weights/best.pt` by mtime, or raise."""
    candidates = list(runs_root.glob("*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"no best.pt found under {runs_root}/*/weights/. "
            "Train first or pass --weights."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def bbox_center_depth(depth_frame, x1: int, y1: int, x2: int, y2: int,
                      width: int, height: int) -> Optional[float]:
    """Depth (m) at the bbox-center pixel; None if 0 / out of range."""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    if not (0 <= cx < width and 0 <= cy < height):
        return None
    d = depth_frame.get_distance(cx, cy)
    if d <= 0.0:
        return None
    return d


def draw_detections(
    img: np.ndarray,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    depth_frame,
) -> np.ndarray:
    """Overlay bbox + 'conf  d.dd m' label for each detection."""
    out = img
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255)
    for (x1f, y1f, x2f, y2f), c in zip(boxes_xyxy, confs):
        x1, y1, x2, y2 = int(round(x1f)), int(round(y1f)), int(round(x2f)), int(round(y2f))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        d = bbox_center_depth(depth_frame, x1, y1, x2, y2, w, h)
        d_str = f"{d:.2f}m" if d is not None else "n/a"
        label = f"{float(c):.2f}  {d_str}"

        ty = max(y1 - 5, 12)
        cv2.putText(out, label, (x1, ty), font, 0.5, color, 2)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(out, (cx, cy), 3, color, -1)
    return out


def draw_fps(img: np.ndarray, fps: Optional[float]) -> None:
    if fps is None:
        return
    cv2.putText(
        img, f"{fps:5.1f} FPS", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Real-time YOLO inference on RealSense D435i color stream.",
    )
    ap.add_argument("--weights", type=Path, default=None,
                    help="path to best.pt (default: newest training/runs/*/weights/best.pt)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence threshold (default 0.25)")
    args = ap.parse_args()

    weights = args.weights or find_latest_best(TRAINING_RUNS)
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    print(f"[realtime] weights: {weights}")
    print(f"[realtime] conf:    {args.conf}")

    model = YOLO(str(weights))

    fps: Optional[float] = None
    last_t: Optional[float] = None

    cv2.namedWindow("realtime", cv2.WINDOW_NORMAL)
    with RealSenseCamera(CAMERA) as camera:
        camera.warmup(num_frames=30)
        while True:
            color, _depth, depth_frame = camera.get_frames()
            if color is None or depth_frame is None:
                continue

            results = model.predict(
                source=color,
                imgsz=IMGSZ,
                conf=args.conf,
                device=DEVICE,
                verbose=False,
                save=False,
                stream=False,
            )
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                color = draw_detections(color, xyxy, confs, depth_frame)

            now = time.perf_counter()
            if last_t is not None:
                inst = 1.0 / max(now - last_t, 1e-6)
                fps = inst if fps is None else (1 - FPS_EMA_ALPHA) * fps + FPS_EMA_ALPHA * inst
            last_t = now
            draw_fps(color, fps)

            cv2.imshow("realtime", color)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
