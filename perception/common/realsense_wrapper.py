"""
RealSense D435i 공용 래퍼
파이프라인 초기화, 프레임 획득 등 모든 모듈에서 공유하는 기능
"""

import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    """RealSense D435i 카메라 관리 클래스"""

    def __init__(self, config):
        """
        Args:
            config: 카메라 설정 dict (CAMERA config)
        """
        self.config = config
        self.pipeline = None
        self.profile = None
        self.align = None
        self.depth_sensor = None
        self.intrinsics = None

    def start(self):
        """파이프라인 시작"""
        self.pipeline = rs.pipeline()
        rs_config = rs.config()

        # 컬러 스트림
        rs_config.enable_stream(
            rs.stream.color,
            self.config["color_width"],
            self.config["color_height"],
            rs.format.bgr8,
            self.config["color_fps"],
        )

        # 깊이 스트림
        rs_config.enable_stream(
            rs.stream.depth,
            self.config["depth_width"],
            self.config["depth_height"],
            rs.format.z16,
            self.config["depth_fps"],
        )

        # IMU 스트림 (VIO에서 사용)
        if self.config.get("enable_imu", False):
            # 주파수를 강제하지 않고, 포맷만 지정하여 SDK가 가능한 값을 찾도록 허용
            rs_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
            rs_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f)

        try:
            self.profile = self.pipeline.start(rs_config)
        except RuntimeError as e:
            if self.config.get("enable_imu", False):
                print(f"[ERROR] 파이프라인 시작 실패 (IMU 충돌/Mac OS 고질적 버그 의심): {e}")
                print("[INFO] 안전 장치 발동: IMU를 끄고 비전 전용(Visual-Only) 모드로 재시도합니다...")
                
                # 새로운 config 생성하고 IMU 제외
                fallback_config = rs.config()
                fallback_config.enable_stream(
                    rs.stream.color, self.config["color_width"], self.config["color_height"],
                    rs.format.bgr8, self.config["color_fps"]
                )
                fallback_config.enable_stream(
                    rs.stream.depth, self.config["depth_width"], self.config["depth_height"],
                    rs.format.z16, self.config["depth_fps"]
                )
                self.profile = self.pipeline.start(fallback_config)
                self.config["enable_imu"] = False  # 상태 업데이트
            else:
                raise e

        # IR 이미터 설정
        device = self.profile.get_device()
        self.depth_sensor = device.first_depth_sensor()
        emitter = 1 if self.config.get("enable_ir_emitter", False) else 0
        self.depth_sensor.set_option(rs.option.emitter_enabled, emitter)

        # Depth → Color 정렬
        if self.config.get("align_depth_to_color", True):
            self.align = rs.align(rs.stream.color)

        # Depth 후처리 필터 초기화
        self.depth_filters = []
        self.decimation_filter = rs.decimation_filter()
        self.decimation_filter.set_option(rs.option.filter_magnitude, 2)

        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        # 캡처 모드와 동일하게 구멍 임의 채우기 끄기 (Data integrity 보존)
        self.spatial_filter.set_option(rs.option.holes_fill, 0)

        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)

        # 캡처 모드와 동일한 colorizer 사용
        self.colorizer = rs.colorizer()

        # hole_filling_filter 제거
        self.depth_filters = [
            self.spatial_filter,
            self.temporal_filter,
        ]

        # 카메라 내부 파라미터 저장
        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        print(f"[INFO] RealSense 파이프라인 시작")
        print(f"  → 컬러: {self.config['color_width']}x{self.config['color_height']} @ {self.config['color_fps']}fps")
        print(f"  → 깊이: {self.config['depth_width']}x{self.config['depth_height']} @ {self.config['depth_fps']}fps")
        print(f"  → Depth 정렬: {'ON' if self.align else 'OFF'}")
        print(f"  → IMU: {'ON' if self.config.get('enable_imu', False) else 'OFF'}")

        return self

    def get_frames(self):
        """
        컬러/깊이 프레임 획득

        Returns:
            color_image: numpy array (BGR)
            depth_image: numpy array (16bit)
            depth_frame: rs.depth_frame 객체
        """
        frames = self.pipeline.wait_for_frames()

        if self.align:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None

        # Depth 후처리 필터 적용
        for f in self.depth_filters:
            depth_frame = f.process(depth_frame).as_depth_frame()

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image, depth_frame

    def get_frames_vio(self):
        """
        VIO용 프레임 획득 (컬러 + 깊이 + IMU + 타임스탬프)

        Returns:
            dict: {
                'color': numpy BGR 이미지,
                'depth': numpy 16bit 깊이 이미지,
                'depth_frame': rs.depth_frame,
                'accel': (x,y,z) 또는 None,
                'gyro': (x,y,z) 또는 None,
                'timestamp': 프레임 타임스탬프 (ms),
            } 또는 None
        """
        frames = self.pipeline.wait_for_frames()
        timestamp = frames.get_timestamp()

        # IMU 데이터 수집
        accel_data = None
        gyro_data = None
        for frame in frames:
            if frame.is_motion_frame():
                motion = frame.as_motion_frame().get_motion_data()
                if frame.get_profile().stream_type() == rs.stream.accel:
                    accel_data = (motion.x, motion.y, motion.z)
                elif frame.get_profile().stream_type() == rs.stream.gyro:
                    gyro_data = (motion.x, motion.y, motion.z)

        # 깊이-컬러 정렬
        if self.align:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None

        # Depth 후처리 필터 적용
        for f in self.depth_filters:
            depth_frame = f.process(depth_frame).as_depth_frame()

        return {
            'color': np.asanyarray(color_frame.get_data()),
            'depth': np.asanyarray(depth_frame.get_data()),
            'depth_frame': depth_frame,
            'accel': accel_data,
            'gyro': gyro_data,
            'timestamp': timestamp,
        }

    def get_imu_data(self, frames):
        """
        IMU 데이터 획득 (VIO용)

        Args:
            frames: RealSense frameset

        Returns:
            accel: (x, y, z) 가속도 데이터
            gyro: (x, y, z) 자이로 데이터
        """
        accel_data = None
        gyro_data = None

        for frame in frames:
            if frame.is_motion_frame():
                motion = frame.as_motion_frame().get_motion_data()
                if frame.get_profile().stream_type() == rs.stream.accel:
                    accel_data = (motion.x, motion.y, motion.z)
                elif frame.get_profile().stream_type() == rs.stream.gyro:
                    gyro_data = (motion.x, motion.y, motion.z)

        return accel_data, gyro_data

    def get_intrinsics(self):
        """카메라 내부 파라미터 반환"""
        return self.intrinsics

    def pixel_to_3d(self, depth_frame, pixel_x, pixel_y):
        """
        2D 픽셀 좌표 + 깊이 → 3D 카메라 좌표 변환

        Args:
            depth_frame: rs.depth_frame
            pixel_x, pixel_y: 픽셀 좌표

        Returns:
            (x, y, z): 카메라 좌표계 3D 위치 (미터)
        """
        depth = depth_frame.get_distance(pixel_x, pixel_y)
        if depth == 0:
            return None
        point_3d = rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [pixel_x, pixel_y], depth
        )
        return point_3d

    def stop(self):
        """파이프라인 종료"""
        if self.pipeline:
            self.pipeline.stop()
            print("[INFO] RealSense 파이프라인 종료")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def warmup(self, num_frames=30):
        """카메라 안정화 대기"""
        print("[INFO] 카메라 안정화 중...")
        for _ in range(num_frames):
            self.pipeline.wait_for_frames()
        print("[INFO] 준비 완료!")


def apply_depth_colormap(depth_image, depth_frame=None, colorizer=None):
    """깊이 이미지에 컬러맵 적용 (시각화용)"""
    if depth_frame is not None and colorizer is not None:
        colorized_frame = colorizer.colorize(depth_frame)
        return np.asanyarray(colorized_frame.get_data())
    else:
        return cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )


def get_depth_distance(depth_frame, x, y):
    """특정 좌표의 깊이값(m) 반환"""
    if depth_frame:
        return depth_frame.get_distance(x, y)
    return 0.0
