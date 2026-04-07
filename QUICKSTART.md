# 건물 Instance Segmentation 파이프라인 - 시작하기 가이드

## 📦 설치 및 초기 설정

### 1단계: 환경 준비

```bash
# Python 3.9 이상 필요
python --version

# 필수 패키지 설치
pip install -r requirements.txt
```

### 2단계: 환경 검증

```bash
python utils.py validate
```

이 명령어는 다음을 확인합니다:
- PyTorch 및 CUDA 설정
- GPU 자동 인식 (RTX 4090 x 4)
- 필수 라이브러리 버전

### 3단계: 초기 설정

```bash
python utils.py setup
```

이 명령어:
- 필요한 디렉토리 생성 (`data/`, `outputs/`, `configs/` 등)
- 기본 설정 파일 생성

### 4단계: 모델 가중치 다운로드

```bash
# YOLOv8 모델 자동 다운로드
python utils.py download --model yolov8m-seg

# 또는 다른 크기 선택
python utils.py download --model yolov8l-seg
```

## 🚀 실행 방법

### 방법 1: 메인 파이프라인 (권장)

```bash
python main.py
```

- `data/` 디렉토리의 모든 GeoTIFF 처리
- 결과를 `outputs/` 디렉토리에 저장
- 기본 설정 사용

### 방법 2: 커스텀 설정으로 실행

```python
from main import BuildingSegmentationPipeline
from src.utils.config import PipelineConfig

# YAML에서 설정 로드
config = PipelineConfig.from_yaml("configs/phase1_yolov8.yaml")

# 또는 프로그래매틱으로 수정
config.model.conf_threshold = 0.5
config.data.tile_size = 512

# 파이프라인 실행
pipeline = BuildingSegmentationPipeline(config)
result = pipeline.process_directory("./data")
```

### 방법 3: 샘플 예제 실행

```bash
# 전체 파이프라인 데모
python examples.py full_pipeline

# 데이터 로더 테스트
python examples.py data_loader

# Multi-GPU 설정 확인
python examples.py multi_gpu

# 설정 관리 데모
python examples.py config
```

## 📊 입력 데이터 형식

### 지원 포맷
- **.tif** / **.tiff** / **.geotiff**: GeoTIFF 파일
- 3-채널 RGB 또는 단채널 이미지
- 좌표계 정보 포함

### 디렉토리 구조
```
data/
├── image1.tif
├── image2.tiff
├── region1.geotiff
└── ...
```

### 메타데이터
Rasterio가 자동으로 다음을 추출합니다:
- CRS (좌표계)
- Transform (지리-픽셀 매핑)
- Bounds (범위)

## 📁 출력 결과

`outputs/` 디렉토리에 생성되는 파일들:

### GeoJSON 형식
```
outputs/
├── image1.geojson      # 웹 GIS 호환
├── image2.geojson
└── ...
```

#### 특징
- 웹 표준 포맷
- Leaflet, Mapbox 등과 호환
- 속성 정보 포함 (면적, 둘레 등)

### Shapefile 형식
```
outputs/
├── image1.shp          # 기하정보
├── image1.shx          # 인덱스
├── image1.dbf          # 속성
├── image1.prj          # 좌표계
└── ...
```

#### 특징
- ArcGIS, QGIS 직접 호환
- 속성 테이블 포함
- 벡터 편집 가능

### JSON 형식
```
outputs/
├── image1.json         # 좌표 및 메트릭
├── image2.json
└── ...
```

#### 구조
```json
[
  {
    "id": 0,
    "coordinates": [[x1, y1], [x2, y2], ...],
    "metrics": {
      "area": 450.5,
      "perimeter": 85.2,
      "width": 25.4,
      "height": 18.2,
      "bbox_aspect_ratio": 1.4,
      "compactness": 1.2
    }
  },
  ...
]
```

### 로그 파일
```
outputs/
└── logs/
    └── building_seg.log
```

## 🔧 설정 커스터마이징

### 성능 튜닝 (configs/phase1_yolov8.yaml)

#### 빠른 처리 (낮은 정확도)
```yaml
model:
  model_name: yolov8s-seg
  conf_threshold: 0.3
  sahi_slice_height: 512
  sahi_slice_width: 512
```

#### 정밀도 우선 (느린 처리)
```yaml
model:
  model_name: yolov8l-seg
  conf_threshold: 0.5
  sahi_slice_height: 1024
  sahi_slice_width: 1024
```

#### 메모리-효율 (VRAM 제약)
```yaml
gpu:
  use_multi_gpu: false
  device_ids: [0]  # 단일 GPU
```

### 후처리 조정

```yaml
postprocessing:
  simplification_tolerance: 2.0  # 단순화 정도 (높을수록 더 단순함)
  right_angle_tolerance: 10.0    # 직각 인식 공차 (도)
  min_polygon_area: 50.0         # 최소 면적 (작을수록 더 많은 객체)
```

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest tests/ -v

# 특정 테스트 실행
pytest tests/test_pipeline.py::TestMaskToPolygon -v

# 커버리지 리포트
pytest tests/ --cov=src --cov-report=html
```

## 📈 성능 최적화 팁

### 1. GPU 메모리 최적화
```bash
# VRAM 사용량 모니터링 (Windows)
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits -l 1
```

### 2. 배치 처리
- 여러 이미지를 한 번에 처리하려면 `process_directory()` 사용
- 자동으로 GPU 메모리 관리

### 3. 타일 크기 최적화
```python
config.data.tile_size = 1024  # VRAM에 따라 조정
config.data.overlap_ratio = 0.1  # 경계 오류 감소
```

## 🔍 디버깅

### 로그 레벨 변경

```python
config.log_level = "DEBUG"  # 더 자세한 정보
```

### 중간 결과 저장

```python
config.save_intermediate = True  # 각 단계의 결과 저장
```

### 단일 파일 테스트

```python
from main import BuildingSegmentationPipeline
from src.utils.config import PipelineConfig

config = PipelineConfig()
pipeline = BuildingSegmentationPipeline(config)
result = pipeline.process_geotiff("path/to/test.tif")
```

## 📚 참고 자료

### 핵심 라이브러리 문서
- [YOLOv8 공식 문서](https://docs.ultralytics.com/)
- [SAHI GitHub](https://github.com/obss/sahi)
- [Rasterio 문서](https://rasterio.readthedocs.io/)
- [Shapely 문서](https://shapely.readthedocs.io/)
- [GeoPandas 문서](https://geopandas.org/)

### 모델 선택 가이드

| 모델 | 속도 | 정확도 | VRAM | 권장용도 |
|------|------|--------|------|---------|
| yolov8n-seg | ⚡⚡⚡ | ⭐⭐ | 2GB | 빠른 테스트 |
| yolov8s-seg | ⚡⚡ | ⭐⭐⭐ | 4GB | 균형잡힌 |
| yolov8m-seg | ⚡ | ⭐⭐⭐⭐ | 8GB | **추천** |
| yolov8l-seg | 🐌 | ⭐⭐⭐⭐⭐ | 12GB | 정밀도 우선 |

## ❓ 자주 묻는 질문

### Q: GPU가 인식되지 않습니다
```bash
python utils.py validate
```
- CUDA 설치 확인
- 드라이버 업데이트 확인

### Q: 메모리 부족 오류
- 타일 크기 감소: `tile_size: 512`
- 단일 GPU 사용: `device_ids: [0]`
- 모델 변경: `yolov8s-seg` 사용

### Q: 결과 품질 향상
- 신뢰도 조정: `conf_threshold: 0.5` (보수적)
- 오버랩 증가: `overlap_ratio: 0.2`
- 모델 업그레이드: `yolov8l-seg`

### Q: GIS 소프트웨어에서 열기
- Shapefile 사용: `QGIS`, `ArcGIS`
- GeoJSON 사용: 온라인 뷰어, 웹GIS

## 🐛 버그 리포팅

이슈 발견 시:
1. 로그 파일 확인: `outputs/logs/building_seg.log`
2. 재현 가능한 최소 예제 준비
3. 환경 정보 수집: `python utils.py validate`

## 📞 연락처

질문이나 피드백: 프로젝트 이슈 트래커 또는 이메일

---

**Happy Building Segmentation! 🏗️**
