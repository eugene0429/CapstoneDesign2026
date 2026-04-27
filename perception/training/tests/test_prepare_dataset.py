from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from perception.training.prepare_dataset import (
    discover_scenarios,
    pair_images_with_labels,
)


def _make_synthetic_dataset(root: Path, scenarios: dict[str, list[tuple[str, str]]]):
    """
    scenarios: { "scenario_01_4m_left": [("img_a", "0 .5 .5 .1 .1"), ("img_b", None), ...] }
    None label means no .txt file. Empty-string label means zero-byte file.
    """
    for scen_dir, items in scenarios.items():
        scen_id = scen_dir.split("_")[1]
        imgs = root / "imgs" / scen_dir
        labs = root / "labels" / f"{scen_id}_labels"
        imgs.mkdir(parents=True)
        labs.mkdir(parents=True)
        for stem, label in items:
            (imgs / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # tiny valid-ish jpg
            if label is not None:
                (labs / f"{stem}.txt").write_text(label)


class TestDiscoverScenarios(unittest.TestCase):
    def test_pairs_image_dirs_with_label_dirs_by_nn_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_synthetic_dataset(root, {
                "scenario_01_4m_left":   [("a", "0 .5 .5 .1 .1")],
                "scenario_02_4m_middle": [("b", "0 .5 .5 .1 .1")],
            })
            result = discover_scenarios(root)
            ids = sorted(r[0] for r in result)
            self.assertEqual(ids, ["01", "02"])
            for sid, imgs, labs in result:
                self.assertTrue(imgs.is_dir())
                self.assertTrue(labs.is_dir())
                self.assertIn(f"scenario_{sid}_", imgs.name)
                self.assertEqual(labs.name, f"{sid}_labels")

    def test_raises_when_label_dir_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "imgs" / "scenario_99_2m_left").mkdir(parents=True)
            # no labels/99_labels
            with self.assertRaises(FileNotFoundError):
                discover_scenarios(root)


class TestPairing(unittest.TestCase):
    def test_skips_images_without_label_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_synthetic_dataset(root, {
                "scenario_01_4m_left": [
                    ("a", "0 .5 .5 .1 .1"),
                    ("b", None),                  # no .txt
                    ("c", "0 .3 .3 .2 .2"),
                ],
            })
            imgs = root / "imgs" / "scenario_01_4m_left"
            labs = root / "labels" / "01_labels"
            pairs = pair_images_with_labels(imgs, labs)
            stems = sorted(p[0].stem for p in pairs)
            self.assertEqual(stems, ["a", "c"])

    def test_skips_zero_byte_label_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_synthetic_dataset(root, {
                "scenario_01_4m_left": [
                    ("a", "0 .5 .5 .1 .1"),
                    ("b", ""),                    # zero-byte .txt
                ],
            })
            imgs = root / "imgs" / "scenario_01_4m_left"
            labs = root / "labels" / "01_labels"
            pairs = pair_images_with_labels(imgs, labs)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][0].stem, "a")


if __name__ == "__main__":
    unittest.main()
