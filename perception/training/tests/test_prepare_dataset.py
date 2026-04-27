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


from perception.training.prepare_dataset import stratified_split


class TestStratifiedSplit(unittest.TestCase):
    def _fake_pairs(self, n: int, prefix: str) -> list:
        return [(Path(f"/tmp/{prefix}_{i}.jpg"), Path(f"/tmp/{prefix}_{i}.txt"))
                for i in range(n)]

    def test_split_sizes_per_scenario_match_ratios(self):
        pairs_by_scenario = {
            "01": self._fake_pairs(100, "s01"),
            "02": self._fake_pairs(100, "s02"),
        }
        splits = stratified_split(pairs_by_scenario, ratios=(0.8, 0.1, 0.1), seed=42)
        # 100 * (0.8, 0.1, 0.1) = (80, 10, 10) per scenario, 200 total split as 160/20/20
        self.assertEqual(len(splits["train"]), 160)
        self.assertEqual(len(splits["val"]), 20)
        self.assertEqual(len(splits["test"]), 20)

    def test_split_handles_uneven_counts_without_dropping(self):
        # 7 items, 0.8/0.1/0.1 -> 5/1/1 (with last bucket taking remainder)
        pairs_by_scenario = {"01": self._fake_pairs(7, "s01")}
        splits = stratified_split(pairs_by_scenario, ratios=(0.8, 0.1, 0.1), seed=42)
        total = sum(len(splits[k]) for k in ("train", "val", "test"))
        self.assertEqual(total, 7)
        self.assertGreaterEqual(len(splits["train"]), 5)
        self.assertGreaterEqual(len(splits["val"]), 1)
        self.assertGreaterEqual(len(splits["test"]), 1)

    def test_split_is_deterministic_with_seed(self):
        pairs_by_scenario = {"01": self._fake_pairs(50, "s01")}
        a = stratified_split(pairs_by_scenario, seed=42)
        b = stratified_split(pairs_by_scenario, seed=42)
        self.assertEqual([p[0].name for p in a["train"]],
                         [p[0].name for p in b["train"]])

    def test_split_changes_with_different_seed(self):
        pairs_by_scenario = {"01": self._fake_pairs(50, "s01")}
        a = stratified_split(pairs_by_scenario, seed=42)
        b = stratified_split(pairs_by_scenario, seed=7)
        self.assertNotEqual([p[0].name for p in a["train"]],
                            [p[0].name for p in b["train"]])

    def test_no_overlap_between_splits(self):
        pairs_by_scenario = {"01": self._fake_pairs(100, "s01")}
        splits = stratified_split(pairs_by_scenario, seed=42)
        train_set = {p[0].name for p in splits["train"]}
        val_set   = {p[0].name for p in splits["val"]}
        test_set  = {p[0].name for p in splits["test"]}
        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set  & test_set), 0)


if __name__ == "__main__":
    unittest.main()
