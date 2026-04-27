"""
Offline dataset augmentation for YOLO26n bell detection.

Reads originals from `data/images/train/` (after `prepare_dataset.py` has
materialised the symlink tree) and writes `multiplier` albumentations-augmented
copies per original to the same dirs. Files are named `<stem>_aug{i}.jpg` /
`<stem>_aug{i}.txt`. Val/test directories are not touched.

Run directly:
    python -m perception.training.augment_dataset                  # multiplier=5, idempotent
    python -m perception.training.augment_dataset --multiplier 10
    python -m perception.training.augment_dataset --rebuild        # wipe *_aug* and regenerate
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np


def _train_dirs(training_root: Path) -> Tuple[Path, Path]:
    return (
        training_root / "data" / "images" / "train",
        training_root / "data" / "labels" / "train",
    )


def list_original_train_pairs(training_root: Path) -> List[Tuple[Path, Path]]:
    """Return [(img, lab), ...] for originals only (no `_aug` in stem).

    Raises FileNotFoundError if the train image dir doesn't exist
    (i.e. prepare_dataset.py hasn't been run yet).
    """
    img_dir, lab_dir = _train_dirs(training_root)
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"train image dir not found: {img_dir} "
            "(run prepare_dataset.py first)"
        )
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if "_aug" in img.stem:
            continue
        lab = lab_dir / f"{img.stem}.txt"
        if not lab.is_file():
            continue
        if lab.stat().st_size == 0:
            continue
        pairs.append((img, lab))
    return pairs


def wipe_augmented(training_root: Path) -> None:
    """Delete every `*_aug*.jpg` / `*_aug*.txt` from train/ image and label dirs."""
    img_dir, lab_dir = _train_dirs(training_root)
    for d in (img_dir, lab_dir):
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if "_aug" in p.stem:
                p.unlink()


def build_transform() -> A.Compose:
    """Build the per-sample albumentations pipeline.

    Each albumentations call samples its own random state internally; deterministic
    seeding for the whole augmentation run is handled in `augment()` by calling
    `np.random.seed(...)` before each transform invocation.
    """
    return A.Compose(
        [
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=70,
                val_shift_limit=40,
                p=1.0,
            ),
            A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.RandomScale(scale_limit=0.1, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(std_range=(0.0, 0.02), p=0.3),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def _read_label_yolo(lab_path: Path) -> Tuple[List[Tuple[float, float, float, float]], List[int]]:
    """Parse a YOLO label file into (bboxes, class_labels)."""
    bboxes: list[tuple[float, float, float, float]] = []
    classes: list[int] = []
    for line in lab_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:5])
        bboxes.append((cx, cy, w, h))
        classes.append(cls)
    return bboxes, classes


def _write_label_yolo(
    lab_path: Path,
    bboxes: List[Tuple[float, float, float, float]],
    classes: List[int],
) -> None:
    if not bboxes:
        lab_path.write_text("")
        return
    lines = [
        f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for cls, (cx, cy, w, h) in zip(classes, bboxes)
    ]
    lab_path.write_text("\n".join(lines) + "\n")


def generate_one(
    img_path: Path,
    lab_path: Path,
    out_img: Path,
    out_lab: Path,
    transform,
    seed: int,
) -> None:
    """Apply `transform` to (img, label) and write to (out_img, out_lab).

    Seeds both global RNGs and (when available) the albumentations Compose's
    internal RNG via `set_random_seed`, so that subsequent runs with the same
    seed produce byte-identical output.
    """
    np.random.seed(seed)
    random.seed(seed)
    # albumentations 2.x maintains its own RNG independent of numpy/random
    # globals; seed it explicitly when the transform exposes the hook.
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)

    image = cv2.imread(str(img_path))
    if image is None:
        raise IOError(f"could not read image: {img_path}")
    bboxes, class_labels = _read_label_yolo(lab_path)

    out = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    cv2.imwrite(str(out_img), out["image"])
    _write_label_yolo(out_lab, list(out["bboxes"]), list(out["class_labels"]))
