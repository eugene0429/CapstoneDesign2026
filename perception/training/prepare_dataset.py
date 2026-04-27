"""
Prepare the YOLO26n bell-detection dataset.

Pairs images in `perception/dataset/imgs/scenario_NN_*/` with labels in
`perception/dataset/labels/NN_labels/`, performs a deterministic stratified
80/10/10 split per scenario, materialises a symlink tree under
`perception/training/data/`, and writes `perception/training/dataset.yaml`
for ultralytics.

Run directly:
    python -m perception.training.prepare_dataset           # idempotent
    python -m perception.training.prepare_dataset --rebuild
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

SCENARIO_RE = re.compile(r"^scenario_(\d+)_")


def discover_scenarios(dataset_root: Path) -> List[Tuple[str, Path, Path]]:
    """Return [(scenario_id, imgs_dir, labels_dir), ...] sorted by scenario_id.

    Raises FileNotFoundError if any image dir's matching label dir is missing.
    """
    imgs_root = dataset_root / "imgs"
    labels_root = dataset_root / "labels"
    out: list[tuple[str, Path, Path]] = []
    for imgs_dir in sorted(p for p in imgs_root.iterdir() if p.is_dir()):
        m = SCENARIO_RE.match(imgs_dir.name)
        if not m:
            continue
        sid = m.group(1)
        labels_dir = labels_root / f"{sid}_labels"
        if not labels_dir.is_dir():
            raise FileNotFoundError(
                f"label dir missing for {imgs_dir.name}: expected {labels_dir}"
            )
        out.append((sid, imgs_dir, labels_dir))
    return out


def pair_images_with_labels(
    imgs_dir: Path, labels_dir: Path
) -> List[Tuple[Path, Path]]:
    """Return [(img_path, label_path), ...] for images with non-empty label txt.

    Images without a `<stem>.txt` or with a zero-byte `<stem>.txt` are dropped.
    """
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(imgs_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lab = labels_dir / f"{img.stem}.txt"
        if not lab.is_file():
            continue
        if lab.stat().st_size == 0:
            continue
        pairs.append((img, lab))
    return pairs
