"""
Perception 모듈 통합 진입점
============================

3가지 모드로 실행 가능:
  python main.py capture     → 데이터 수집
  python main.py vio         → VIO 측위 테스트
  python main.py detect      → 타겟 디텍션 + 3D 위치 추정
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="RealSense D435i Perception 모듈",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
모드 설명:
  capture   RealSense 카메라로 YOLO 학습 데이터 수집
  vio       Visual-Inertial Odometry 기반 실시간 측위
  detect    타겟 디텍션 + 깊이 기반 3D 위치 추정
        """,
    )
    parser.add_argument(
        "mode",
        choices=["capture", "vio", "detect"],
        help="실행 모드 선택",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="YOLO 모델 경로 (detect 모드에서 사용)",
    )
    parser.add_argument(
        "--no-imu", action="store_true",
        help="IMU를 끄고 Visual-Only 모드로 실행 (vio 모드 전용)",
    )

    args, remaining = parser.parse_known_args()

    if args.mode == "capture":
        from data_collection.capture import RealsenseCapture, parse_args
        # 남은 인자를 capture의 argparse에 전달
        sys.argv = [sys.argv[0]] + remaining
        capture_args = parse_args()
        capture = RealsenseCapture(capture_args)
        capture.start()

    elif args.mode == "vio":
        from vio.vio_runner import run_vio
        run_vio(use_imu=(not args.no_imu))

    elif args.mode == "detect":
        print("[DETECT] 타겟 디텍션 + 3D 위치 추정 모드")
        print("[TODO] 디텍션 파이프라인 구현 필요")
        print("  → detection/detector.py, detection/position_estimator.py 참조")


if __name__ == "__main__":
    main()
