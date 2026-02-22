"""
사전학습된 YOLOv8 모델을 사용한 즉시 테스트 스크립트
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json

from settings import (
    IMAGE_DATA_PATH,
    ANNOTATION_DATA_PATH,
    RESULTS_DIR,
    YOLO_MODEL,
    YOLO_CONFIDENCE,
    CLASS_NAMES,
    COLORS
)
from app_utils.xml_parser import (
    parse_xml_annotation,
    get_annotation_files,
    get_image_path_from_annotation
)
from app_utils.visualization_utils import (
    draw_bboxes,
    create_comparison_image,
    save_detection_results
)


def load_yolo_model(model_name: str = YOLO_MODEL):
    """
    사전학습된 YOLO 모델 로드

    Args:
        model_name: YOLO 모델 이름 (예: 'yolov8n.pt')

    Returns:
        YOLO 모델 객체
    """
    print(f"YOLO 모델 로딩 중: {model_name}")
    model = YOLO(model_name)
    print("모델 로딩 완료!")
    return model


def detect_objects(model, image_path: Path, conf_threshold: float = YOLO_CONFIDENCE):
    """
    이미지에서 객체 탐지

    Args:
        model: YOLO 모델
        image_path: 이미지 경로
        conf_threshold: 신뢰도 임계값

    Returns:
        탐지 결과 리스트
    """
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"이미지 로드 실패: {image_path}")
        return None, []

    # YOLO 추론
    results = model(image, conf=conf_threshold, verbose=False)

    # 결과 파싱
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # COCO 클래스 이름 (YOLOv8 사전학습 모델은 COCO 데이터셋 기반)
            class_name = result.names[cls]

            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'class': class_name,
                'conf': conf,
                'class_id': cls
            })

    return image, detections


def test_on_sample_images(model, num_samples: int = 10, date: str = '20201216'):
    """
    샘플 이미지에 대해 테스트 수행

    Args:
        model: YOLO 모델
        num_samples: 테스트할 샘플 수
        date: 테스트할 날짜
    """
    print(f"\n{'='*60}")
    print(f"샘플 이미지 테스트 시작 (날짜: {date}, 샘플 수: {num_samples})")
    print(f"{'='*60}\n")

    # 어노테이션 파일 가져오기
    annotation_files = get_annotation_files(ANNOTATION_DATA_PATH, date)

    if len(annotation_files) == 0:
        print(f"어노테이션 파일을 찾을 수 없습니다: {ANNOTATION_DATA_PATH / date}")
        return

    print(f"총 {len(annotation_files)}개의 어노테이션 파일 발견")

    # 샘플 선택
    sample_files = annotation_files[:num_samples]

    # 결과 저장 디렉토리
    test_results_dir = RESULTS_DIR / "test_samples" / date
    test_results_dir.mkdir(parents=True, exist_ok=True)

    results_summary = []

    for idx, xml_path in enumerate(tqdm(sample_files, desc="테스트 진행")):
        # Ground Truth 파싱
        gt_data = parse_xml_annotation(str(xml_path))

        # 이미지 경로 찾기
        image_path = get_image_path_from_annotation(xml_path, IMAGE_DATA_PATH)

        if not image_path.exists():
            print(f"\n이미지 파일 없음: {image_path}")
            continue

        # 객체 탐지
        image, detections = detect_objects(model, image_path)

        if image is None:
            continue

        # Ground Truth 시각화
        gt_detections = []
        for obj in gt_data['objects']:
            gt_detections.append({
                'bbox': obj['bbox'],
                'class': obj['name'],
                'conf': 1.0
            })

        gt_image = draw_bboxes(image.copy(), gt_detections, COLORS, show_conf=False)

        # 예측 결과 시각화
        pred_image = draw_bboxes(image.copy(), detections, COLORS, show_conf=True)

        # 비교 이미지 생성
        comparison = create_comparison_image(image, gt_image, pred_image)

        # 저장
        save_path = test_results_dir / f"sample_{idx:03d}_{xml_path.stem}.jpg"
        cv2.imwrite(str(save_path), comparison)

        # 결과 요약
        results_summary.append({
            'image': str(image_path.name),
            'ground_truth_count': len(gt_detections),
            'prediction_count': len(detections),
            'predictions': detections
        })

        # 콘솔 출력
        print(f"\n[샘플 {idx+1}/{num_samples}] {image_path.name}")
        print(f"  Ground Truth: {len(gt_detections)}개 객체")
        print(f"  예측 결과: {len(detections)}개 객체")
        if detections:
            for det in detections:
                print(f"    - {det['class']}: {det['conf']:.2f}")

    # 결과 요약 저장
    summary_path = test_results_dir / "results_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"테스트 완료!")
    print(f"결과 저장 위치: {test_results_dir}")
    print(f"요약 파일: {summary_path}")
    print(f"{'='*60}\n")


def main():
    """
    메인 실행 함수
    """
    print("="*60)
    print("해양 선박 및 부유물 감지 - 사전학습 모델 테스트")
    print("="*60)

    # YOLO 모델 로드
    model = load_yolo_model()

    # 샘플 이미지 테스트
    test_on_sample_images(model, num_samples=20, date='20201216')

    print("\n테스트가 완료되었습니다!")
    print(f"결과는 {RESULTS_DIR / 'test_samples'} 폴더에서 확인하실 수 있습니다.")


if __name__ == "__main__":
    main()
