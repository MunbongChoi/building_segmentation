"""프로젝트 설정 관리"""
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    """데이터 관련 설정"""
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    
    # GeoTIFF 타일 설정
    tile_size: int = 1024
    overlap_ratio: float = 0.1  # SAHI 오버랩
    
    # 데이터 포맷
    input_format: str = "tif"  # tif, tiff, geotiff
    

@dataclass
class ModelConfig:
    """모델 관련 설정"""
    # Phase 1: YOLOv8-Seg 설정
    model_name: str = "yolov8m-seg"  # yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg
    model_weights: str = "yolov8m-seg.pt"  # 또는 custom 가중치 경로
    
    # 신뢰도 임계값
    conf_threshold: float = 0.4
    iou_threshold: float = 0.5
    
    # SAHI 설정
    sahi_slice_height: int = 1024
    sahi_slice_width: int = 1024
    sahi_overlap_height_ratio: float = 0.1
    sahi_overlap_width_ratio: float = 0.1
    
    # 후처리 NMS
    mask_nms_threshold: float = 0.5
    

@dataclass
class GPUConfig:
    """GPU 병렬 처리 설정"""
    device_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])  # 4x RTX 4090
    use_multi_gpu: bool = True
    batch_size: int = 1  # SAHI는 일반적으로 배치 처리 미지원
    num_workers: int = 4
    pin_memory: bool = True
    

@dataclass
class PostProcessingConfig:
    """후처리 설정"""
    # Douglas-Peucker 단순화
    simplification_tolerance: float = 1.0  # 픽셀 단위
    
    # 직각화 설정
    enable_right_angle_regularization: bool = True
    right_angle_tolerance: float = 5.0  # 도(degree)
    
    # 최소 폴리곤 면적 (픽셀^2)
    min_polygon_area: float = 100.0
    
    # 출력 포맷
    output_formats: List[str] = field(default_factory=lambda: ["geojson", "shapefile", "json"])
    

@dataclass
class PipelineConfig:
    """전체 파이프라인 설정"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    postprocessing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    
    # 로깅
    log_level: str = "INFO"
    save_intermediate: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """YAML 파일에서 설정 로드"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f) or {}
        
        data_dict = config_dict.get('data', {}) or {}
        if 'data_dir' in data_dict:
            data_dict['data_dir'] = Path(data_dict['data_dir'])
        if 'output_dir' in data_dict:
            data_dict['output_dir'] = Path(data_dict['output_dir'])
        
        return cls(
            data=DataConfig(**data_dict),
            model=ModelConfig(**(config_dict.get('model', {}) or {})),
            gpu=GPUConfig(**(config_dict.get('gpu', {}) or {})),
            postprocessing=PostProcessingConfig(**(config_dict.get('postprocessing', {}) or {})),
            log_level=config_dict.get('log_level', "INFO"),
            save_intermediate=config_dict.get('save_intermediate', True),
        )
    
    def to_yaml(self, yaml_path: str) -> None:
        """설정을 YAML 파일로 저장"""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            'data': {
                **self.data.__dict__,
                'data_dir': str(self.data.data_dir),
                'output_dir': str(self.data.output_dir),
            },
            'model': self.model.__dict__,
            'gpu': self.gpu.__dict__,
            'postprocessing': self.postprocessing.__dict__,
            'log_level': self.log_level,
            'save_intermediate': self.save_intermediate,
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


# 기본 설정 인스턴스
DEFAULT_CONFIG = PipelineConfig()
