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

from pathlib import Path
from typing import List, Tuple


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
