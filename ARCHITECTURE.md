# 🏗️ 건물 Instance Segmentation 파이프라인 - 아키텍처 개요

## 📋 프로젝트 완성 요약

**Phase 1 (YOLOv8-Seg + SAHI + Multi-GPU) 완전 구현**

### ✅ 완성된 컴포넌트

#### 1. **핵심 추론 엔진**
- 📁 `src/core/`
  - `dataset.py`: GeoTIFF 로더 & 타일 기반 처리
  - `model_manager.py`: Multi-GPU 모델 관리 (4x RTX 4090)
  - `inference.py`: SAHI 기반 대규모 이미지 추론

#### 2. **후처리 파이프라인**
- 📁 `src/postprocessing/`
  - `mask_to_polygon.py`: 픽셀 마스크 → 폴리곤 변환
    - Douglas-Peucker 단순화
    - 직각화 (Right-Angle Regularization)
    - 최소 면적 필터링
  - `geometry_utils.py`: 벡터 파일 생성
    - ✅ GeoJSON (웹 표준)
    - ✅ Shapefile (GIS 소프트웨어)
    - ✅ JSON (메트릭 포함)

#### 3. **설정 & 로깅**
- 📁 `src/utils/`
  - `config.py`: YAML 기반 설정 관리
  - `logger.py`: 구조화된 로깅 (로테이션 지원)

#### 4. **통합 파이프라인**
- `main.py`: 전체 엔드-투-엔드 파이프라인
  - `process_geotiff()`: 단일 파일 처리
  - `process_directory()`: 배치 처리

#### 5. **유틸리티 & 예제**
- `utils.py`: 환경 설정 및 검증
  - setup: 초기 디렉토리 생성
  - validate: 환경 점검
  - download: 모델 가중치 다운로드
  - cleanup: 출력 정리
- `examples.py`: 실행 가능한 예제 코드

#### 6. **테스트 & 문서**
- `tests/test_pipeline.py`: 단위 테스트
- `README.md`: 프로젝트 전체 문서
- `QUICKSTART.md`: 빠른 시작 가이드
- `requirements.txt`: 의존성 명시

### 📊 주요 기능

#### 🎯 데이터 처리 흐름

```
GeoTIFF 파일 로드
    ├─ 지리정보 추출 (좌표, CRS, Transform)
    └─ 이미지 데이터 정규화
          ↓
타일 분할 (SAHI)
    ├─ Sliding Window: 1024x1024 (설정 가능)
    ├─ 오버랩: 10% (경계 오류 감소)
    └─ 타일별 인덱싱
          ↓
YOLOv8-Seg 추론 (Multi-GPU)
    ├─ GPU 0,1,2,3 자동 할당
    ├─ 인스턴스 마스크 생성
    ├─ 신뢰도 필터링
    └─ GPU 메모리 최적화
          ↓
마스크 병합 및 NMS
    ├─ 전역 좌표계 매핑
    ├─ 중복 제거 (마스크 IOU 기반)
    └─ 경계 정제
          ↓
마스크 → 폴리곤 변환
    ├─ OpenCV 외곽선 감지
    ├─ Douglas-Peucker 단순화
    ├─ 직각화 (선택)
    └─ 최소 면적 필터
          ↓
벡터 파일 생성
    ├─ GeoJSON (웹 호환)
    ├─ Shapefile (GIS 호환)
    └─ JSON (메트릭)
```

#### ⚙️ 설정 시스템

```yaml
# 데이터 설정
data:
  tile_size: 1024              # 타일 크기
  overlap_ratio: 0.1           # 오버랩 비율

# 모델 설정
model:
  model_name: yolov8m-seg      # YOLOv8 크기
  conf_threshold: 0.4          # 신뢰도
  sahi_slice_height: 1024      # SAHI 슬라이스 높이

# GPU 설정
gpu:
  device_ids: [0, 1, 2, 3]    # RTX 4090 x 4
  use_multi_gpu: true

# 후처리
postprocessing:
  simplification_tolerance: 1.0
  enable_right_angle_regularization: true
  right_angle_tolerance: 5.0
  min_polygon_area: 100.0
```

#### 🚀 실행 방법

**1. 기본 실행**
```bash
python main.py
```

**2. 환경 검증**
```bash
python utils.py validate
```

**3. 초기 설정**
```bash
python utils.py setup
python utils.py download --model yolov8m-seg
```

**4. 예제 실행**
```bash
python examples.py full_pipeline
python examples.py config
```

**5. 테스트**
```bash
pytest tests/ -v
```

### 📈 성능 특성

#### Multi-GPU 병렬 처리
- **GPU 할당**: 자동 감지 (최대 4개)
- **메모리 관리**: 타일 기반 처리로 VRAM 최적화
- **캐시 정리**: 자동 GPU 메모리 해제

#### 처리 속도 (예상)
- **YOLOv8m-Seg**: 약 100-200ms/이미지 (1024x1024)
- **타일 분할**: 대규모 이미지도 1-2분 내 처리
- **후처리**: 추론 시간의 약 10-20%

#### 메모리 사용
- **YOLOv8m**: 약 8GB/GPU
- **타일 기반**: 풀사이즈 이미지도 처리 가능
- **최대 활용**: 4090 x 4 = 96GB 병렬 처리 능력

### 🔄 디렉토리 구조

```
building_segmentation/
├── src/                          # 소스 코드
│   ├── core/
│   │   ├── dataset.py           # GeoTIFF 로더
│   │   ├── model_manager.py     # Multi-GPU 관리
│   │   └── inference.py         # SAHI 엔진
│   ├── postprocessing/
│   │   ├── mask_to_polygon.py   # 마스크→폴리곤
│   │   └── geometry_utils.py    # 벡터 파일
│   └── utils/
│       ├── config.py            # 설정
│       └── logger.py            # 로깅
├── configs/
│   └── phase1_yolov8.yaml       # 설정 파일
├── data/                        # 입력 GeoTIFF
├── outputs/                     # 출력 결과
│   ├── *.geojson
│   ├── *.shp, *.shx, *.dbf
│   ├── *.json
│   └── logs/
├── tests/
│   └── test_pipeline.py
├── main.py                      # 메인 파이프라인
├── examples.py                  # 예제 코드
├── utils.py                     # 유틸리티
├── requirements.txt             # 의존성
├── README.md                    # 전체 문서
├── QUICKSTART.md                # 빠른 시작
└── ARCHITECTURE.md              # 이 파일
```

### 🔧 커스터마이징 예시

#### 모델 변경 (Phase 1 내)
```python
config.model.model_name = "yolov8l-seg"  # 더 정밀함
# 또는
config.model.model_name = "yolov8s-seg"  # 더 빠름
```

#### 타일 크기 조정
```python
config.data.tile_size = 512   # 메모리 절감
config.data.overlap_ratio = 0.2  # 경계 오류 감소
```

#### 직각화 설정
```python
config.postprocessing.enable_right_angle_regularization = True
config.postprocessing.right_angle_tolerance = 5.0  # 도 단위
```

### 📦 의존성 요약

**핵심 라이브러리**
- `torch`, `torchvision`: 딥러닝 프레임워크
- `ultralytics`: YOLOv8 구현
- `sahi`: Slicing Aided Hyper Inference
- `rasterio`: GeoTIFF 처리
- `geopandas`, `shapely`: 지리정보 벡터 처리
- `opencv-python`: 이미지 처리

**버전**: 모두 최신 안정 버전 지정 완료

### 🎯 Phase 2 준비 (다음 단계)

현재 Phase 1이 완성되었으므로, Phase 2로의 전환은 다음과 같이 진행됩니다:

#### Phase 2 아키텍처 (계획)
```python
# Phase 2: Mask2Former + OBB
from src.core.model_manager_phase2 import Mask2FormerManager

config.model.model_name = "mask2former"
config.model.backbone = "swin-l"  # Swin-Large
config.model.use_obb = True        # OBB 지원
```

#### 호환성 관리
- 모든 후처리 로직 **재사용 가능**
- 설정 시스템 **확장 가능**
- 파이프라인 인터페이스 **일관성 유지**

### 💾 출력 파일 형식

#### GeoJSON
```json
{
  "type": "FeatureCollection",
  "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
  "features": [
    {
      "type": "Feature",
      "id": 0,
      "geometry": {"type": "Polygon", "coordinates": [...]},
      "properties": {"area": 450.5, "perimeter": 85.2}
    }
  ]
}
```

#### JSON (메트릭)
```json
[
  {
    "id": 0,
    "coordinates": [[x1,y1], [x2,y2], ...],
    "metrics": {
      "area": 450.5,
      "perimeter": 85.2,
      "width": 25.4,
      "height": 18.2,
      "bbox_aspect_ratio": 1.4,
      "compactness": 1.2
    }
  }
]
```

#### Shapefile
- `*.shp`: 기하정보
- `*.shx`: 인덱스
- `*.dbf`: 속성 테이블
- `*.prj`: 좌표계 (EPSG:4326)

### 🧪 테스트 커버리지

- ✅ 데이터 로더 (타일 생성)
- ✅ 마스크→폴리곤 변환
- ✅ 직각화 로직
- ✅ 지오메트리 메트릭
- ✅ 설정 관리 (YAML I/O)
- ✅ 통합 파이프라인

### 📝 로깅 시스템

**로그 레벨**: DEBUG, INFO, WARNING, ERROR, CRITICAL

**저장 위치**: `outputs/logs/building_seg.log`

**특징**:
- 자동 로테이션 (100MB마다)
- 7일 보관
- 콘솔 + 파일 동시 출력
- 컬러 포매팅 지원

---

## 🚀 시작하기

```bash
# 1. 설정
python utils.py setup

# 2. 모델 다운로드
python utils.py download --model yolov8m-seg

# 3. 실행
python main.py

# 4. 결과 확인
# outputs/ 디렉토리의 GeoJSON, Shapefile 파일 확인
```

---

**Project Version**: 0.1.0 (Phase 1)  
**Status**: ✅ Production Ready  
**Last Updated**: 2026-04-07
