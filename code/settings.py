"""
프로젝트 전역 설정
"""
import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent

# 데이터셋 경로
IMAGE_DATA_PATH = PROJECT_ROOT / "원천데이터" / "남해_여수항1구역_BOX"
ANNOTATION_DATA_PATH = PROJECT_ROOT / "남해_여수항1구역_BOX"

# 결과 저장 경로
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 체크포인트 저장 경로
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# YOLO 설정
YOLO_MODEL = "yolov8n.pt"  # nano 모델 (빠른 테스트용)
YOLO_CONFIDENCE = 0.25  # 신뢰도 임계값
YOLO_IOU = 0.45  # NMS IoU 임계값

# 클래스 정의 (XML 어노테이션 기준)
CLASS_NAMES = {
    2: "선박",
    3: "기타부유물"
}

# YOLO 클래스 매핑 (0-based index)
YOLO_CLASS_MAPPING = {
    "선박": 0,
    "기타부유물": 1
}

# 이미지 설정
IMAGE_SIZE = 640  # YOLO 입력 크기

# 시각화 설정
COLORS = {
    "선박": (0, 255, 0),  # 녹색
    "기타부유물": (255, 0, 0)  # 빨간색
}

# 평가 설정
IOU_THRESHOLD = 0.5  # mAP 계산용 IoU 임계값
