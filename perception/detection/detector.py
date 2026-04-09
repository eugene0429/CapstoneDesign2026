"""
YOLO 기반 타겟 디텍션
=====================

학습된 YOLO 모델을 사용하여 실시간 객체 탐지 수행

TODO:
  - YOLO 모델 로딩 및 추론
  - 바운딩 박스 + 클래스 + confidence 반환
  - NMS 후처리
"""

import numpy as np


class TargetDetector:
    """YOLO 기반 객체 탐지기"""

    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Args:
            model_path: YOLO 모델 가중치 파일 경로 (.pt)
            confidence_threshold: 최소 confidence 임계값
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None

    def load_model(self):
        """YOLO 모델 로드"""
        # TODO: ultralytics YOLO 모델 로딩
        # from ultralytics import YOLO
        # self.model = YOLO(self.model_path)
        raise NotImplementedError("YOLO 모델 로딩 구현 필요")

    def detect(self, color_image):
        """
        이미지에서 객체 탐지 수행

        Args:
            color_image: BGR 컬러 이미지

        Returns:
            detections: list of dict
                - bbox: (x1, y1, x2, y2) 바운딩 박스
                - class_id: 클래스 인덱스
                - class_name: 클래스 이름
                - confidence: 탐지 신뢰도
        """
        # TODO: 구현
        raise NotImplementedError("객체 탐지 구현 필요")

    def get_bbox_center(self, bbox):
        """바운딩 박스 중심점 반환"""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
