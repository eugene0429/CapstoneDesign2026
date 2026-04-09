"""
녹화된 비디오에서 프레임 추출
YOLO 학습용 이미지 생성

사용법:
  python extract_frames.py --video dataset/videos/video.mp4
  python extract_frames.py --video dataset/videos/video.mp4 --interval 0.5
  python extract_frames.py --video dataset/videos/video.mp4 --interval 0 --every 10
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from config import PATHS


def extract_frames(video_path, output_dir=None, interval=1.0, every_n=None, prefix="frame"):
    """
    비디오에서 프레임 추출

    Args:
        video_path: 입력 비디오 경로
        output_dir: 출력 디렉터리 (기본: dataset/images)
        interval: 추출 간격 (초). every_n이 지정되면 무시됨
        every_n: N 프레임마다 추출. None이면 interval 사용
        prefix: 파일명 접두사
    """
    if output_dir is None:
        output_dir = PATHS["images"]
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 비디오를 열 수 없습니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"[INFO] 비디오 정보:")
    print(f"  → 파일: {video_path}")
    print(f"  → FPS: {fps:.1f}")
    print(f"  → 총 프레임: {total_frames}")
    print(f"  → 재생 시간: {duration:.1f}초")
    print()

    if every_n:
        frame_interval = every_n
        print(f"[INFO] 매 {every_n} 프레임마다 추출")
    else:
        frame_interval = max(1, int(fps * interval))
        print(f"[INFO] {interval}초 간격으로 추출 (매 {frame_interval} 프레임)")

    saved_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = f"{prefix}_{frame_idx:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

            # 진행률 표시
            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            print(f"\r[EXTRACT] {progress:5.1f}% | {saved_count}장 추출됨", end="", flush=True)

        frame_idx += 1

    cap.release()
    print(f"\n\n[완료] {saved_count}장의 프레임을 추출했습니다.")
    print(f"  → 저장 경로: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="녹화 비디오에서 YOLO 학습용 프레임 추출")
    parser.add_argument("--video", required=True, help="입력 비디오 파일 경로")
    parser.add_argument("--output", default=None, help="출력 디렉터리 (기본: dataset/images)")
    parser.add_argument("--interval", type=float, default=1.0, help="추출 간격 (초). 기본: 1.0")
    parser.add_argument("--every", type=int, default=None, help="N 프레임마다 추출. 지정 시 interval 무시")
    parser.add_argument("--prefix", default="frame", help="파일명 접두사. 기본: frame")

    args = parser.parse_args()
    extract_frames(args.video, args.output, args.interval, args.every, args.prefix)
