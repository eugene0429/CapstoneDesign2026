"""
유틸리티 함수 모음
RealSense 파이프라인 초기화, 프레임 처리, 파일 저장 등
"""

import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs

from config import CAMERA, PATHS, CAPTURE, DISPLAY

# ---------------------------------------------------------
# RealSense 후처리 필터 (Post-Processing Filters) 전역 선언 (지연 초기화)
# ---------------------------------------------------------
spatial_filter = None
temporal_filter = None
colorizer = None

def init_filters():
    global spatial_filter, temporal_filter, colorizer
    if spatial_filter is None:
        spatial_filter = rs.spatial_filter()
        spatial_filter.set_option(rs.option.filter_magnitude, 2)
        spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        # 데이터 정합성(Data Integrity) 확보를 위해 임의로 구멍을 채우지 않음(0 처리)
        spatial_filter.set_option(rs.option.holes_fill, 0)
        
        temporal_filter = rs.temporal_filter()
        # Data Integrity 우선을 위해 가짜 픽셀을 만드는 hole_filling_filter 제거
        colorizer = rs.colorizer()



def create_directories():
    """데이터셋 저장 디렉터리 생성"""
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
    print("[INFO] 디렉터리 생성 완료:")
    for name, path in PATHS.items():
        print(f"  → {name}: {path}")


def init_realsense_pipeline():
    """
    RealSense 파이프라인 초기화
    Returns:
        pipeline: rs.pipeline 객체
        profile: 스트리밍 프로파일
        align: rs.align 객체 (depth→color 정렬)
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # 컬러 스트림 설정
    config.enable_stream(
        rs.stream.color,
        CAMERA["color_width"],
        CAMERA["color_height"],
        rs.format.bgr8,
        CAMERA["color_fps"],
    )

    # 깊이 스트림 설정
    config.enable_stream(
        rs.stream.depth,
        CAMERA["depth_width"],
        CAMERA["depth_height"],
        rs.format.z16,
        CAMERA["depth_fps"],
    )

    # 파이프라인 시작
    profile = pipeline.start(config)

    # 센서 및 파이프라인이 준비된 후 후처리 필터 초기화
    init_filters()

    # IR 이미터 설정
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    if CAMERA["enable_ir_emitter"]:
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
    else:
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

    # Depth → Color 정렬 객체
    align = rs.align(rs.stream.color) if CAMERA["align_depth_to_color"] else None

    print(f"[INFO] RealSense 파이프라인 시작")
    print(f"  → 컬러: {CAMERA['color_width']}x{CAMERA['color_height']} @ {CAMERA['color_fps']}fps")
    print(f"  → 깊이: {CAMERA['depth_width']}x{CAMERA['depth_height']} @ {CAMERA['depth_fps']}fps")
    print(f"  → Depth 정렬: {'ON' if align else 'OFF'}")

    return pipeline, profile, align


def get_frames(pipeline, align=None):
    """
    파이프라인에서 컬러/깊이 프레임 획득
    Returns:
        color_image: numpy array (BGR)
        depth_image: numpy array (16bit)
        depth_frame: rs.depth_frame 객체
    """
    frames = pipeline.wait_for_frames()

    if align:
        frames = align.process(frames)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return None, None, None

    # [필터 적용] 품질 개선을 위해 정렬된 Depth Frame에 후처리 필터를 적용합니다.
    depth_frame = spatial_filter.process(depth_frame)
    depth_frame = temporal_filter.process(depth_frame)
    # Hole filling 과정을 제외하여 0값(가려짐, 측정불가 영역)을 검은색으로 정직하게 보존합니다.
    depth_frame = depth_frame.as_depth_frame()

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    return color_image, depth_image, depth_frame


def apply_depth_colormap(depth_image, depth_frame=None):
    """깊이 이미지에 컬러맵 적용 (시각화용)"""
    if depth_frame is not None:
        # Intel에서 권장하는 colorizer 적용 (노이즈 최소화, 자동 스케일링)
        colorized_frame = colorizer.colorize(depth_frame)
        return np.asanyarray(colorized_frame.get_data())
    else:
        # Fallback 구조 (이전 방식)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            DISPLAY.get("depth_colormap", cv2.COLORMAP_JET),
        )
        return depth_colormap


def save_image(color_image, depth_image, prefix="img"):
    """
    컬러 + 깊이 이미지 저장
    Returns:
        filename: 저장된 파일 이름 (확장자 제외)
    """
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}"

    # 컬러 이미지 저장
    color_path = os.path.join(PATHS["images"], filename + CAPTURE["image_format"])
    params = [cv2.IMWRITE_JPEG_QUALITY, CAPTURE["image_quality"]]
    cv2.imwrite(color_path, color_image, params)

    # 깊이 이미지 저장 (16bit PNG)
    depth_path = os.path.join(PATHS["depth"], filename + CAPTURE["depth_format"])
    cv2.imwrite(depth_path, depth_image)

    return filename


def draw_info_overlay(frame, info_dict, recording=False):
    """프레임 위에 정보 오버레이 그리기"""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 반투명 상단 바
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y_offset = 20
    for key, value in info_dict.items():
        text = f"{key}: {value}"
        cv2.putText(
            frame, text, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            DISPLAY["font_scale"],
            DISPLAY["font_color"],
            DISPLAY["font_thickness"],
            cv2.LINE_AA,
        )
        y_offset += 20

    # 녹화 중 표시
    if recording:
        cv2.circle(frame, (w - 25, 15), 8, (0, 0, 255), -1)
        cv2.putText(
            frame, "REC", (w - 60, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA,
        )

    # 하단 조작 안내
    help_text = "[S] Save | [R] Record | [A] Auto | [D] Depth | [Q] Quit"
    cv2.putText(
        frame, help_text, (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )

    return frame


def get_depth_distance(depth_frame, x, y):
    """특정 좌표의 깊이값(m) 반환"""
    if depth_frame:
        return depth_frame.get_distance(x, y)
    return 0.0
