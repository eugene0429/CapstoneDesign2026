"""
RealSense D435i - YOLO 학습 데이터 캡처 도구
=============================================

기능:
  [S] 수동 캡처 - 현재 프레임을 이미지로 저장
  [A] 자동 캡처 - 설정된 간격으로 자동 저장 (토글)
  [R] 비디오 녹화 - MP4 영상 녹화 시작/중지 (토글)
  [D] 깊이 뷰 - 깊이 이미지 표시 토글
  [+/-] 해상도 변경 - 캡처 해상도 전환
  [Q] 종료

사용법:
  python capture.py
  python capture.py --no-depth         # 깊이 저장 없이 컬러만
  python capture.py --auto 1.0         # 자동 캡처 간격 1초
  python capture.py --prefix obj       # 파일명 접두사 지정
  python capture.py --resolution 1280  # 해상도 지정
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CAMERA, PATHS, CAPTURE, DISPLAY
from data_collection.utils import (
    create_directories,
    init_realsense_pipeline,
    get_frames,
    apply_depth_colormap,
    save_image,
    draw_info_overlay,
    get_depth_distance,
)


class RealsenseCapture:
    """RealSense D435i 캡처 컨트롤러"""

    def __init__(self, args):
        self.args = args
        self.pipeline = None
        self.profile = None
        self.align = None

        # 상태 변수
        self.capture_count = 0
        self.is_recording = False
        self.is_auto_capture = False
        self.show_depth = DISPLAY["show_depth"]
        self.video_writer = None
        self.last_auto_time = 0
        self.auto_interval = args.auto if args.auto else CAPTURE["auto_interval"]
        self.prefix = args.prefix
        self.start_time = time.time()

    def start(self):
        """캡처 세션 시작"""
        print("=" * 60)
        print("  RealSense D435i - YOLO 학습 데이터 캡처 도구")
        print("=" * 60)

        # 디렉터리 생성
        create_directories()

        # 파이프라인 초기화
        try:
            self.pipeline, self.profile, self.align = init_realsense_pipeline()
        except Exception as e:
            print(f"\n[ERROR] RealSense 카메라 연결 실패: {e}")
            print("  → 카메라가 USB 3.0 포트에 연결되어 있는지 확인하세요.")
            print("  → 다른 프로그램이 카메라를 사용 중인지 확인하세요.")
            sys.exit(1)

        # 카메라 안정화 대기
        print("\n[INFO] 카메라 안정화 중...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("[INFO] 준비 완료! 캡처를 시작하세요.\n")

        self._print_controls()
        self._main_loop()

    def _print_controls(self):
        """조작법 출력"""
        print("┌─────────────────────────────────────┐")
        print("│           조작 키 안내               │")
        print("├─────────────────────────────────────┤")
        print("│  [S]     수동 캡처 (이미지 저장)     │")
        print("│  [A]     자동 캡처 ON/OFF            │")
        print("│  [R]     비디오 녹화 시작/중지        │")
        print("│  [D]     깊이 뷰 ON/OFF              │")
        print("│  [Q]     종료                        │")
        print("└─────────────────────────────────────┘")
        print()

    def _main_loop(self):
        """메인 캡처 루프"""
        try:
            while True:
                # 프레임 획득
                color_image, depth_image, depth_frame = get_frames(
                    self.pipeline, self.align
                )

                if color_image is None:
                    continue

                # 자동 캡처 처리
                if self.is_auto_capture:
                    now = time.time()
                    if now - self.last_auto_time >= self.auto_interval:
                        self._save_current(color_image, depth_image)
                        self.last_auto_time = now

                # 비디오 녹화 처리
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(color_image)

                # 디스플레이 프레임 준비
                display_frame = self._build_display(
                    color_image, depth_image, depth_frame
                )

                # 화면 출력
                cv2.imshow(DISPLAY["window_name"], display_frame)

                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key, color_image, depth_image):
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C 감지 - 종료합니다.")
        finally:
            self._cleanup()

    def _build_display(self, color_image, depth_image, depth_frame):
        """디스플레이 프레임 구성"""
        display = color_image.copy()

        # 중앙 십자선 + 깊이 표시
        h, w = display.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.drawMarker(
            display, (cx, cy), (0, 255, 0),
            cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA
        )
        center_dist = get_depth_distance(depth_frame, cx, cy)

        # 정보 오버레이
        elapsed = time.time() - self.start_time
        info = {
            "Captured": f"{self.capture_count}",
            "Center Depth": f"{center_dist:.2f}m",
            "Auto": f"ON ({self.auto_interval}s)" if self.is_auto_capture else "OFF",
            "Time": f"{int(elapsed)}s",
        }
        display = draw_info_overlay(display, info, self.is_recording)

        # 깊이 뷰 결합
        if self.show_depth:
            depth_colormap = apply_depth_colormap(depth_image, depth_frame)
            # 깊이 뷰를 컬러 뷰의 1/3 크기로 축소하여 우측 하단에 표시
            small_h, small_w = h // 3, w // 3
            depth_small = cv2.resize(depth_colormap, (small_w, small_h))

            # 테두리 추가
            cv2.rectangle(depth_small, (0, 0), (small_w - 1, small_h - 1), (255, 255, 255), 1)

            # 우측 하단에 오버레이
            y1 = h - small_h - 10
            x1 = w - small_w - 10
            display[y1:y1 + small_h, x1:x1 + small_w] = depth_small

        return display

    def _handle_key(self, key, color_image, depth_image):
        """
        키 입력 처리
        Returns: False면 루프 종료
        """
        if key == ord('q') or key == ord('Q'):
            return False

        elif key == ord('s') or key == ord('S'):
            self._save_current(color_image, depth_image)

        elif key == ord('a') or key == ord('A'):
            self.is_auto_capture = not self.is_auto_capture
            self.last_auto_time = time.time()
            state = "ON" if self.is_auto_capture else "OFF"
            print(f"[AUTO] 자동 캡처 {state} (간격: {self.auto_interval}s)")

        elif key == ord('r') or key == ord('R'):
            self._toggle_recording(color_image)

        elif key == ord('d') or key == ord('D'):
            self.show_depth = not self.show_depth
            state = "ON" if self.show_depth else "OFF"
            print(f"[DEPTH] 깊이 뷰 {state}")

        return True

    def _save_current(self, color_image, depth_image):
        """현재 프레임 저장"""
        filename = save_image(color_image, depth_image, self.prefix)
        self.capture_count += 1
        print(f"[SAVE] #{self.capture_count:04d} → {filename}")

    def _toggle_recording(self, color_image):
        """비디오 녹화 토글"""
        if not self.is_recording:
            # 녹화 시작
            timestamp = int(time.time())
            video_path = os.path.join(
                PATHS["videos"],
                f"{self.prefix}_video_{timestamp}{CAPTURE['video_format']}"
            )
            h, w = color_image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*CAPTURE["video_codec"])
            self.video_writer = cv2.VideoWriter(
                video_path, fourcc, CAPTURE["video_fps"], (w, h)
            )
            self.is_recording = True
            print(f"[REC] 녹화 시작 → {video_path}")
        else:
            # 녹화 중지
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            print("[REC] 녹화 중지")

    def _cleanup(self):
        """리소스 정리"""
        print("\n[INFO] 정리 중...")

        if self.is_recording and self.video_writer:
            self.video_writer.release()
            print("[INFO] 비디오 저장 완료")

        if self.pipeline:
            self.pipeline.stop()
            print("[INFO] RealSense 파이프라인 종료")

        cv2.destroyAllWindows()

        print(f"\n[결과] 총 {self.capture_count}장의 이미지를 캡처했습니다.")
        print(f"  → 이미지: {PATHS['images']}")
        print(f"  → 깊이:   {PATHS['depth']}")
        print(f"  → 비디오: {PATHS['videos']}")
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="RealSense D435i YOLO 학습 데이터 캡처 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-depth", action="store_true",
        help="깊이 이미지 저장 비활성화",
    )
    parser.add_argument(
        "--auto", type=float, default=None,
        help=f"자동 캡처 간격 (초). 기본값: {CAPTURE['auto_interval']}",
    )
    parser.add_argument(
        "--prefix", type=str, default="img",
        help="저장 파일명 접두사. 기본값: img",
    )
    parser.add_argument(
        "--resolution", type=int, choices=[640, 1280, 1920], default=None,
        help="캡처 해상도 (너비). 기본값: config.py 설정 사용",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 해상도 오버라이드
    if args.resolution:
        if args.resolution == 640:
            CAMERA["color_width"], CAMERA["color_height"] = 640, 480
            CAMERA["depth_width"], CAMERA["depth_height"] = 640, 480
        elif args.resolution == 1280:
            CAMERA["color_width"], CAMERA["color_height"] = 1280, 720
            CAMERA["depth_width"], CAMERA["depth_height"] = 1280, 720
        elif args.resolution == 1920:
            CAMERA["color_width"], CAMERA["color_height"] = 1920, 1080
            # D435i 깊이 센서의 최대 지원 해상도는 1280x720
            CAMERA["depth_width"], CAMERA["depth_height"] = 1280, 720

    capture = RealsenseCapture(args)
    capture.start()
