# Building Instance Segmentation Pipeline - Phase 1

대용량 위성/항공 정사영상에서 건물을 탐지하고 픽셀 단위 정밀 마스크를 추출하는 Instance Segmentation 파이프라인입니다.

## 📋 프로젝트 개요

### Phase 1: YOLOv8-Seg + SAHI + Multi-GPU
- **목표:** 파이프라인 검증 및 PoC
- **모델:** YOLOv8 Instance Segmentation
- **기법:** SAHI (Slicing Aided Hyper Inference)
- **하드웨어:** RTX 4090 x 4 Multi-GPU

### Phase 2: 정밀도 극대화 (계획)
- Mask2Former 또는 PointRend 기반 모델 적용
- OBB (Oriented Bounding Box) 지원
- 고정밀 건물 경계선 추출

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Python 3.9+ 필요
pip install -r requirements.txt
```

### 2. 설정 파일 생성

```python
from src.utils.config import PipelineConfig

config = PipelineConfig()
config.to_yaml("configs/phase1_yolov8.yaml")
```

### 3. 파이프라인 실행

```bash
# 메인 파이프라인
python main.py

# 또는 커스텀 설정 사용
python main.py --config configs/phase1_yolov8.yaml
```

## 📁 프로젝트 구조

```
building_segmentation/
├── src/
│   ├── core/
│   │   ├── dataset.py          # GeoTIFF 로더 및 타일링
│   │   ├── model_manager.py    # Multi-GPU 모델 관리
│   │   └── inference.py        # SAHI 기반 추론 엔진
│   ├── postprocessing/
│   │   ├── mask_to_polygon.py  # 마스크→폴리곤 변환
│   │   └── geometry_utils.py   # GeoJSON/Shapefile 생성
│   └── utils/
│       ├── config.py           # 설정 관리
│       └── logger.py           # 로깅
├── configs/
│   └── phase1_yolov8.yaml      # 설정 파일
├── data/                        # 입력 데이터
├── outputs/                     # 출력 결과
├── tests/                       # 테스트
├── main.py                      # 메인 엔트리 포인트
└── requirements.txt             # 의존성
```

## 🔧 핵심 기능

### 1. 데이터 로딩 (GeoTIFF)
- Rasterio 기반 GeoTIFF 파일 로드
- 지리정보 (좌표, CRS) 자동 관리
- 메타데이터 보존

### 2. 타일 기반 처리
- Sliding window 방식 타일 분할 (기본 1024x1024)
- 타일 간 오버랩 설정 가능 (기본 10%)
- 전역 좌표계 매핑

### 3. SAHI 기반 추론
- 대용량 이미지의 효율적 처리
- 슬라이스별 병렬 추론 가능
- 마스크 기반 NMS (중복 제거)

### 4. 후처리
- 마스크 → 폴리곤 변환
- Douglas-Peucker 알고리즘으로 단순화
- 직각화 (Right-Angle Regularization)
- 최소 면적 필터링

### 5. 벡터 파일 생성
- **GeoJSON:** 웹 및 GIS 소프트웨어 호환
- **Shapefile:** ArcGIS, QGIS 호환
- **JSON:** 좌표 및 메트릭 저장 (COCO 형식 선택가능)

## ⚙️ 설정 예시

```yaml
data:
  data_dir: ./data
  output_dir: ./outputs
  tile_size: 1024
  overlap_ratio: 0.1

model:
  model_name: yolov8m-seg
  model_weights: yolov8m-seg.pt
  conf_threshold: 0.4
  iou_threshold: 0.5
  sahi_slice_height: 1024
  sahi_slice_width: 1024
  sahi_overlap_height_ratio: 0.1
  sahi_overlap_width_ratio: 0.1

gpu:
  device_ids: [0, 1, 2, 3]
  use_multi_gpu: true
  batch_size: 1
  num_workers: 4

postprocessing:
  simplification_tolerance: 1.0
  enable_right_angle_regularization: true
  right_angle_tolerance: 5.0
  min_polygon_area: 100.0
  output_formats: [geojson, shapefile, json]
```

## 📊 출력 형식

### GeoJSON
```json
{
  "type": "FeatureCollection",
  "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
  "features": [
    {
      "type": "Feature",
      "id": 0,
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[x1, y1], [x2, y2], ...]]
      },
      "properties": {
        "id": 0,
        "area": 450.5,
        "perimeter": 85.2
      }
    }
  ]
}
```

### Shapefile
- QGIS, ArcGIS 등과 직접 호환
- 속성 테이블: area, perimeter, id 등

### JSON
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
  }
]
```

## 🔄 처리 흐름

```
GeoTIFF 입력
    ↓
타일 분할 (1024x1024, 10% 오버랩)
    ↓
각 타일에서 YOLOv8-Seg 추론
    ↓
마스크 수집 및 전역 좌표계 병합
    ↓
NMS로 중복 제거
    ↓
마스크 → 폴리곤 변환
    ↓
Douglas-Peucker 단순화
    ↓
직각화 (선택사항)
    ↓
벡터 파일 저장 (GeoJSON, Shapefile, JSON)
```

## 📈 성능 최적화

### Multi-GPU 활용
- 4개의 RTX 4090 자동 감지 및 사용
- 각 디바이스별 메모리 모니터링
- 동적 배치 크기 조정 (계획)

### 메모리 효율화
- 타일 기반 처리로 메모리 사용 최소화
- GPU 캐시 자동 정리
- 필요시 CPU-GPU 메모리 교환

## 🧪 테스트

```bash
pytest tests/ -v
```

## 📝 로그

로그는 `outputs/logs/` 디렉토리에 저장됩니다:
- 콘솔 출력 (INFO 이상)
- 파일 저장 (DEBUG 이상)
- 자동 로테이션 (100MB마다)

## 🔮 다음 단계

### Phase 2 계획
1. Mask2Former 모델 통합
2. OBB 지원 (대각선 건물)
3. 성능 벤치마킹
4. 웹 시각화 대시보드

### 개선 사항
- [ ] 동적 배치 처리
- [ ] 분산 처리 (Ray, Dask)
- [ ] 실시간 모니터링
- [ ] 모델 앙상블
- [ ] TensorRT 최적화

## 📞 참고자료

- [YOLOv8 문서](https://docs.ultralytics.com/)
- [SAHI GitHub](https://github.com/obss/sahi)
- [Rasterio 문서](https://rasterio.readthedocs.io/)
- [Shapely 문서](https://shapely.readthedocs.io/)
- [GeoPandas 문서](https://geopandas.org/)

## 📄 라이선스

MIT License

---

**작성자:** MunbongChoi  
**생성일:** 2026-04-07  
**버전:** 0.1.0 (Phase 1 - PoC)
