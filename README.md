# 해양 선박 및 부유물 감지 시스템

RAFT 기반 모션 벡터와 YOLO를 결합한 해상 환경 객체 감지 시스템

## 빠른 시작 (즉시 테스트)

### 1. 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. 사전학습 모델로 즉시 테스트

```bash
python test.py
```

이 명령어는:
- 사전학습된 YOLOv8 모델을 자동으로 다운로드
- 샘플 이미지 20장에 대해 객체 탐지 수행
- Ground Truth와 예측 결과를 비교한 이미지 생성
- 결과를 `results/test_samples/` 폴더에 저장

### 3. 결과 확인

```
results/test_samples/20201216/
├── sample_000_여수항_맑음_20201216_0014_0001.jpg
├── sample_001_여수항_맑음_20201216_0014_0002.jpg
├── ...
└── results_summary.json
```

각 이미지는 3개 패널로 구성:
- **Original**: 원본 이미지
- **Ground Truth**: 실제 라벨링 (녹색: 선박, 빨간색: 부유물)
- **Prediction**: YOLO 예측 결과

## 프로젝트 구조

```
metaBTS/
├── config.py                    # 전역 설정
├── requirements.txt             # 의존성
├── test.py                      # 즉시 테스트 스크립트
├── utils/
│   ├── xml_parser.py           # XML 어노테이션 파싱
│   └── visualization_utils.py  # 시각화 도구
├── results/                     # 결과 저장
├── 원천데이터/                  # 이미지 데이터
└── 남해_여수항1구역_BOX/        # 어노테이션 데이터
```

## 주요 기능

### Phase 1: 기본 인식률 확인 (현재)
- ✅ 사전학습 YOLOv8 모델 사용
- ✅ XML 어노테이션 파싱
- ✅ 객체 탐지 및 시각화
- ✅ Ground Truth 비교

### Phase 2: RAFT 기반 개선 (예정)
- ⬜ GMC (Global Motion Compensation)
- ⬜ RAFT 광학 흐름 추출
- ⬜ 모션 벡터 기반 물보라 필터링
- ⬜ 시공간 특징 분석

## 설정 변경

`config.py`에서 다음 설정을 변경할 수 있습니다:

```python
YOLO_MODEL = "yolov8n.pt"      # 모델 크기 (n/s/m/l/x)
YOLO_CONFIDENCE = 0.25         # 신뢰도 임계값
```

## 데이터셋 정보

- **이미지**: 3840x2160 해상도
- **클래스**: 선박, 기타부유물
- **형식**: JPG 이미지 + XML 어노테이션

## 문제 해결

### 이미지를 찾을 수 없는 경우
`config.py`에서 경로 확인:
```python
IMAGE_DATA_PATH = PROJECT_ROOT / "원천데이터" / "남해_여수항1구역_BOX"
ANNOTATION_DATA_PATH = PROJECT_ROOT / "남해_여수항1구역_BOX"
```

### GPU 사용
CUDA가 설치되어 있으면 자동으로 GPU 사용

## 다음 단계

1. `test.py` 실행 결과 확인
2. 인식률 분석
3. 필요시 RAFT 기반 개선 진행
