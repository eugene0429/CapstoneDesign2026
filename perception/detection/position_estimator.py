"""
타겟 3D 위치 추정
=================

디텍션 결과 (2D bbox) + 깊이 데이터 + VIO 포즈를 결합하여
타겟의 월드 좌표계 3D 위치를 추정

파이프라인:
  1. YOLO로 타겟 bbox 탐지
  2. bbox 중심의 깊이값 획득
  3. 2D 픽셀 + 깊이 → 3D 카메라 좌표 변환 (deprojection)
  4. VIO 포즈로 카메라 좌표 → 월드 좌표 변환

TODO:
  - 깊이 필터링 (bbox 영역 내 유효 깊이 중앙값 사용)
  - 카메라 → 월드 좌표 변환
  - 다중 프레임 위치 평균/필터링
"""

import numpy as np


class PositionEstimator:
    """타겟 3D 위치 추정기"""

    def __init__(self, camera):
        """
        Args:
            camera: RealSenseCamera 인스턴스 (common.realsense_wrapper)
        """
        self.camera = camera

    def estimate(self, detection, depth_frame, camera_pose=None):
        """
        단일 탐지 결과의 3D 월드 좌표 추정

        Args:
            detection: dict with 'bbox' key (x1, y1, x2, y2)
            depth_frame: rs.depth_frame 객체
            camera_pose: 4x4 변환 행렬 (VIO로부터, None이면 카메라 좌표만 반환)

        Returns:
            position_3d: (x, y, z) 월드 좌표 (미터), 또는 None
        """
        # bbox 중심 좌표
        bbox = detection["bbox"]
        cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)

        # 카메라 좌표계 3D 위치
        point_camera = self.camera.pixel_to_3d(depth_frame, cx, cy)
        if point_camera is None:
            return None

        # VIO 포즈가 있으면 월드 좌표로 변환
        if camera_pose is not None:
            point_camera_h = np.array([*point_camera, 1.0])
            point_world = camera_pose @ point_camera_h
            return tuple(point_world[:3])

        return tuple(point_camera)

    def estimate_batch(self, detections, depth_frame, camera_pose=None):
        """
        다중 탐지 결과의 3D 위치 일괄 추정

        Args:
            detections: list of detection dicts
            depth_frame: rs.depth_frame
            camera_pose: 4x4 변환 행렬

        Returns:
            results: list of dict (detection + position_3d)
        """
        results = []
        for det in detections:
            pos = self.estimate(det, depth_frame, camera_pose)
            result = {**det, "position_3d": pos}
            results.append(result)
        return results
