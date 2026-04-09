"""
YOLO 학습 데이터셋 구조 정리
캡처된 이미지를 YOLO 학습용 폴더 구조로 변환

YOLO 데이터셋 구조:
  dataset/
  ├── data.yaml          ← 데이터셋 설정 파일
  ├── train/
  │   ├── images/
  │   └── labels/
  └── val/
      ├── images/
      └── labels/

사용법:
  python organize_dataset.py
  python organize_dataset.py --ratio 0.8
  python organize_dataset.py --classes person car dog
"""

import argparse
import os
import random
import shutil
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PATHS, YOLO


def organize(train_ratio=0.8, classes=None, seed=42):
    """
    캡처된 이미지를 YOLO 데이터셋 구조로 정리

    Args:
        train_ratio: 학습 데이터 비율 (0.0~1.0)
        classes: 클래스 목록
        seed: 랜덤 시드
    """
    random.seed(seed)

    source_images = PATHS["images"]
    source_labels = PATHS["labels"]
    dataset_dir = PATHS["dataset"]

    # 이미지 파일 목록
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [
        f for f in os.listdir(source_images)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not images:
        print("[ERROR] 이미지가 없습니다. 먼저 capture.py로 이미지를 캡처하세요.")
        return

    print(f"[INFO] 총 {len(images)}장의 이미지를 발견했습니다.")

    # 셔플 및 분할
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    print(f"[INFO] Train: {len(train_images)}장 | Val: {len(val_images)}장")

    # YOLO 디렉터리 구조 생성
    dirs = {
        "train_images": os.path.join(dataset_dir, "train", "images"),
        "train_labels": os.path.join(dataset_dir, "train", "labels"),
        "val_images": os.path.join(dataset_dir, "val", "images"),
        "val_labels": os.path.join(dataset_dir, "val", "labels"),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 파일 복사
    def copy_files(file_list, img_dst, lbl_dst):
        copied = 0
        for img_file in file_list:
            # 이미지 복사
            src = os.path.join(source_images, img_file)
            dst = os.path.join(img_dst, img_file)
            shutil.copy2(src, dst)

            # 라벨 파일 복사 (있는 경우)
            label_name = os.path.splitext(img_file)[0] + ".txt"
            label_src = os.path.join(source_labels, label_name)
            if os.path.exists(label_src):
                label_dst = os.path.join(lbl_dst, label_name)
                shutil.copy2(label_src, label_dst)
                copied += 1

        return copied

    print("\n[COPY] 파일 복사 중...")
    train_labels = copy_files(train_images, dirs["train_images"], dirs["train_labels"])
    val_labels = copy_files(val_images, dirs["val_images"], dirs["val_labels"])

    print(f"  → Train: {len(train_images)}장 이미지, {train_labels}개 라벨")
    print(f"  → Val: {len(val_images)}장 이미지, {val_labels}개 라벨")

    # data.yaml 생성
    if classes is None:
        classes = YOLO.get("classes", [])

    if not classes:
        print("\n[WARNING] 클래스가 지정되지 않았습니다.")
        print("  → data.yaml의 'names' 필드를 직접 수정하세요.")
        print("  → 또는 --classes 옵션으로 클래스를 지정하세요.")
        classes = ["class_0"]

    data_yaml = {
        "path": os.path.abspath(dataset_dir),
        "train": "train/images",
        "val": "val/images",
        "nc": len(classes),
        "names": classes,
    }

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\n[YAML] 데이터셋 설정 파일 생성: {yaml_path}")
    print(f"  → 클래스 수: {len(classes)}")
    print(f"  → 클래스 목록: {classes}")

    # 결과 요약
    print("\n" + "=" * 50)
    print("  데이터셋 정리 완료!")
    print("=" * 50)
    print(f"\n데이터셋 구조:")
    print(f"  {dataset_dir}/")
    print(f"  ├── data.yaml")
    print(f"  ├── train/")
    print(f"  │   ├── images/ ({len(train_images)}장)")
    print(f"  │   └── labels/ ({train_labels}개)")
    print(f"  └── val/")
    print(f"      ├── images/ ({len(val_images)}장)")
    print(f"      └── labels/ ({val_labels}개)")

    print(f"\n[NEXT] 라벨링 도구를 사용해 annotations을 생성하세요:")
    print(f"  → CVAT: https://www.cvat.ai/")
    print(f"  → Roboflow: https://roboflow.com/")
    print(f"  → LabelImg: pip install labelImg")

    if train_labels == 0 and val_labels == 0:
        print(f"\n[TIP] 아직 라벨 파일이 없습니다.")
        print(f"  → dataset/labels/ 폴더에 YOLO 형식 .txt 파일을 추가한 후")
        print(f"  → 이 스크립트를 다시 실행하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 데이터셋 구조 정리")
    parser.add_argument(
        "--ratio", type=float, default=0.8,
        help="Train 비율 (0.0~1.0). 기본: 0.8"
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="클래스 이름 목록. 예: --classes person car dog"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드. 기본: 42"
    )

    args = parser.parse_args()
    organize(args.ratio, args.classes, args.seed)
