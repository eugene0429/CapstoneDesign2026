from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from perception.training.augment_dataset import (
    list_original_train_pairs,
    wipe_augmented,
)


def _seed_train_dir(training_root: Path,
                    originals: list[str],
                    augmented: list[str] = ()) -> None:
    """Create the data/{images,labels}/train/ skeleton with given stems."""
    img_dir = training_root / "data" / "images" / "train"
    lab_dir = training_root / "data" / "labels" / "train"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    for stem in originals:
        (img_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (lab_dir / f"{stem}.txt").write_text("0 .5 .5 .1 .1")
    for stem in augmented:
        (img_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (lab_dir / f"{stem}.txt").write_text("0 .5 .5 .1 .1")


class TestListOriginalTrainPairs(unittest.TestCase):
    def test_returns_only_pairs_without_aug_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_train_dir(tr,
                originals=["img_001", "img_002"],
                augmented=["img_001_aug0", "img_002_aug3"])
            pairs = list_original_train_pairs(tr)
            stems = sorted(p[0].stem for p in pairs)
            self.assertEqual(stems, ["img_001", "img_002"])

    def test_pairs_are_image_label_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_train_dir(tr, originals=["img_a"])
            pairs = list_original_train_pairs(tr)
            self.assertEqual(len(pairs), 1)
            img, lab = pairs[0]
            self.assertEqual(img.suffix, ".jpg")
            self.assertEqual(lab.suffix, ".txt")
            self.assertEqual(img.stem, lab.stem)

    def test_raises_when_train_dir_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                list_original_train_pairs(Path(tmp))


class TestWipeAugmented(unittest.TestCase):
    def test_deletes_aug_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_train_dir(tr,
                originals=["img_001"],
                augmented=["img_001_aug0", "img_001_aug1"])
            wipe_augmented(tr)
            img_dir = tr / "data" / "images" / "train"
            lab_dir = tr / "data" / "labels" / "train"
            remaining_imgs = sorted(p.name for p in img_dir.iterdir())
            remaining_labs = sorted(p.name for p in lab_dir.iterdir())
            self.assertEqual(remaining_imgs, ["img_001.jpg"])
            self.assertEqual(remaining_labs, ["img_001.txt"])

    def test_idempotent_when_no_aug_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_train_dir(tr, originals=["img_001"])
            wipe_augmented(tr)  # should not raise
            img_dir = tr / "data" / "images" / "train"
            self.assertTrue((img_dir / "img_001.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
