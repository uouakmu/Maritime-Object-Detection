"""
시각화 유틸리티
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path


def draw_bboxes(image: np.ndarray, 
                detections: List[Dict], 
                color_map: Dict[str, Tuple[int, int, int]] = None,
                thickness: int = 2,
                show_conf: bool = True) -> np.ndarray:
    """
    이미지에 바운딩 박스 그리기
    
    Args:
        image: 원본 이미지 (BGR)
        detections: 탐지 결과 리스트 [{'bbox': [x1,y1,x2,y2], 'class': str, 'conf': float}, ...]
        color_map: 클래스별 색상 매핑
        thickness: 선 두께
        show_conf: 신뢰도 표시 여부
        
    Returns:
        바운딩 박스가 그려진 이미지
    """
    img_copy = image.copy()
    
    if color_map is None:
        color_map = {
            '선박': (0, 255, 0),
            '기타부유물': (255, 0, 0)
        }
    
    for det in detections:
        bbox = det['bbox']
        class_name = det['class']
        conf = det.get('conf', 1.0)
        
        x1, y1, x2, y2 = map(int, bbox)
        color = color_map.get(class_name, (255, 255, 255))
        
        # 바운딩 박스 그리기
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        # 레이블 텍스트
        if show_conf:
            label = f"{class_name} {conf:.2f}"
        else:
            label = class_name
        
        # 텍스트 배경
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            img_copy,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # 텍스트
        cv2.putText(
            img_copy,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return img_copy


def create_comparison_image(original: np.ndarray,
                           ground_truth: np.ndarray,
                           prediction: np.ndarray) -> np.ndarray:
    """
    원본, Ground Truth, 예측 결과를 나란히 배치한 비교 이미지 생성
    
    Args:
        original: 원본 이미지
        ground_truth: Ground Truth 바운딩 박스가 그려진 이미지
        prediction: 예측 바운딩 박스가 그려진 이미지
        
    Returns:
        3개 이미지를 가로로 연결한 이미지
    """
    # 이미지 크기 조정 (높이 통일)
    h = min(original.shape[0], ground_truth.shape[0], prediction.shape[0])
    
    original_resized = cv2.resize(original, (int(original.shape[1] * h / original.shape[0]), h))
    gt_resized = cv2.resize(ground_truth, (int(ground_truth.shape[1] * h / ground_truth.shape[0]), h))
    pred_resized = cv2.resize(prediction, (int(prediction.shape[1] * h / prediction.shape[0]), h))
    
    # 레이블 추가
    def add_title(img, title):
        img_with_title = np.zeros((h + 40, img.shape[1], 3), dtype=np.uint8)
        img_with_title[40:, :] = img
        cv2.putText(img_with_title, title, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img_with_title
    
    original_titled = add_title(original_resized, "Original")
    gt_titled = add_title(gt_resized, "Ground Truth")
    pred_titled = add_title(pred_resized, "Prediction")
    
    # 가로로 연결
    comparison = np.hstack([original_titled, gt_titled, pred_titled])
    
    return comparison


def save_detection_results(image: np.ndarray,
                          detections: List[Dict],
                          save_path: Path,
                          color_map: Dict = None):
    """
    탐지 결과를 이미지로 저장
    
    Args:
        image: 원본 이미지
        detections: 탐지 결과
        save_path: 저장 경로
        color_map: 색상 매핑
    """
    result_img = draw_bboxes(image, detections, color_map)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), result_img)
