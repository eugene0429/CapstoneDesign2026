"""
VIO 실시간 실행 루프
RealSense D435i 카메라로 VIO 측위를 실행하고 결과를 시각화
"""

import cv2
import numpy as np
import time

from config import CAMERA, VIO as VIO_CONFIG
from common.realsense_wrapper import RealSenseCamera, apply_depth_colormap


def draw_overlay(image, tracker, fps):
    """VIO 상태 오버레이 표시"""
    stats = tracker.get_stats()
    pos = tracker.get_position()
    euler = tracker.get_euler_degrees()

    lines = [
        f"FPS: {fps:.1f}",
        f"Features: {stats['tracked_features']} | Inliers: {stats['inliers']}",
        f"Keyframe age: {stats['frames_since_keyframe']}",
        f"Pos: X={pos[0]:.3f} Y={pos[1]:.3f} Z={pos[2]:.3f} m",
        f"Rot: R={euler[0]:.1f} P={euler[1]:.1f} Y={euler[2]:.1f} deg",
    ]

    for i, line in enumerate(lines):
        y = 25 + i * 25
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA)

    return image


def draw_trajectory(traj_image, positions, scale=100, size=400):
    """2D 궤적 (XZ 평면, top-down view) 그리기"""
    traj_image[:] = 40  # 어두운 배경

    center = np.array([size // 2, size // 2])

    if len(positions) < 2:
        return traj_image

    for i in range(1, len(positions)):
        p0 = positions[i - 1]
        p1 = positions[i]
        # XZ 평면 투영
        pt0 = (int(center[0] + p0[0] * scale), int(center[1] - p0[2] * scale))
        pt1 = (int(center[0] + p1[0] * scale), int(center[1] - p1[2] * scale))
        cv2.line(traj_image, pt0, pt1, (0, 255, 0), 1, cv2.LINE_AA)

    # 현재 위치
    cur = positions[-1]
    cur_pt = (int(center[0] + cur[0] * scale), int(center[1] - cur[2] * scale))
    cv2.circle(traj_image, cur_pt, 4, (0, 0, 255), -1)

    # 축 표시
    cv2.putText(traj_image, "X", (size - 20, size // 2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    cv2.putText(traj_image, "Z", (size // 2 + 5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    cv2.putText(traj_image, "Trajectory (top-down)", (5, size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

    return traj_image


def run_vio(use_imu=True):
    """VIO 메인 루프"""
    cam_config = {**CAMERA, "enable_imu": use_imu}

    if use_imu:
        print("[VIO] Visual-Inertial Odometry (VIO) 모드로 시작 (IMU 켜짐)")
    else:
        print("[VIO] Visual-Only Odometry 모드로 시작 (IMU 꺼짐)")

    # 카메라 하드웨어 리셋 후 안정화 대기
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) > 0:
        print("[VIO] 카메라 하드웨어 리셋 중...")
        devices[0].hardware_reset()
        time.sleep(5)

    camera = RealSenseCamera(cam_config)
    camera.start()
    camera.warmup(30)

    from vio.vio_tracker import VIOTracker
    tracker = VIOTracker(camera.get_intrinsics(), VIO_CONFIG)

    # 궤적 기록
    positions = []
    traj_size = 400
    traj_image = np.zeros((traj_size, traj_size, 3), dtype=np.uint8)

    print("[VIO] 실행 중... 'q' 키로 종료, 'r' 키로 리셋")

    frame_count = 0
    fps = 0.0
    t_start = time.time()

    try:
        while True:
            data = camera.get_frames_vio()
            if data is None:
                continue

            pose = tracker.update(
                color_image=data['color'],
                depth_image=data['depth'],
                accel=data['accel'],
                gyro=data['gyro'],
                timestamp=data['timestamp'],
            )

            # FPS 계산
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            # 궤적 기록
            pos = tracker.get_position()
            positions.append(pos.copy())
            if len(positions) > 2000:
                positions = positions[-1000:]

            # 시각화
            display = data['color'].copy()
            draw_overlay(display, tracker, fps)

            # 추적 중인 특징점 표시
            if tracker.prev_points is not None and len(tracker.prev_points) > 0:
                for pt in tracker.prev_points:
                    cv2.circle(display, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

            # 궤적 그리기
            draw_trajectory(traj_image, positions, scale=100, size=traj_size)

            # 깊이 컬러맵 (캡처 모드처럼 rs.colorizer 적용)
            depth_color = apply_depth_colormap(
                data['depth'], 
                depth_frame=data.get('depth_frame'), 
                colorizer=camera.colorizer
            )

            # 레이아웃: [카메라 뷰 | 깊이 | 궤적]
            depth_resized = cv2.resize(depth_color, (traj_size, traj_size))
            display_resized = cv2.resize(display, (int(traj_size * display.shape[1] / display.shape[0]), traj_size))

            combined = np.hstack([display_resized, depth_resized, traj_image])
            cv2.imshow("VIO Tracker", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker.reset()
                positions.clear()
                print("[VIO] 리셋 완료")

    except KeyboardInterrupt:
        print("\n[VIO] 종료 (Ctrl+C)")
    finally:
        cv2.destroyAllWindows()
        camera.stop()
        print("[VIO] 종료 완료")
