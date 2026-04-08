"""프로젝트 설정 관리"""
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List
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
    device: str = "cuda"  # cuda, cpu, auto
    require_cuda: bool = True
    allow_cpu_fallback: bool = False
    cuda_visible_devices: str = ""  # 예: "0,1,2,3". 비우면 환경값 유지
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
class AugmentationConfig:
    """추론/학습 augmentation 설정

    현재 실행 파이프라인에는 inference-time augmentation(TTA)만 적용된다.
    training 딕셔너리는 향후 학습 루프 추가 시 같은 YAML을 재사용하기 위한 보존 영역이다.
    """
    enable_tta: bool = False
    tta_scales: List[float] = field(default_factory=lambda: [1.0])
    horizontal_flip: bool = False
    vertical_flip: bool = False
    merge_iou_threshold: float = 0.5
    training: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """YOLOv8-Seg fine-tuning settings.

    The training path expects the Ultralytics segmentation label format:
    class_id x1 y1 x2 y2 ... with normalized polygon coordinates.
    """
    enabled: bool = False
    dataset_yaml: str = ""
    train_images: str = "data/images/train"
    val_images: str = "data/images/val"
    train_labels: str = "data/labels/train"
    val_labels: str = "data/labels/val"
    use_train_as_val_if_missing: bool = True
    class_names: List[str] = field(default_factory=lambda: ["building"])

    pretrained_weights: str = ""
    project: str = "outputs/training"
    name: str = "yolov8_building_seg"
    exist_ok: bool = True
    resume: bool = False

    epochs: int = 100
    imgsz: int = 1024
    batch: int = 8
    workers: int = 8
    patience: int = 30
    device: str = ""  # empty means derive from gpu.device_ids
    seed: int = 42
    deterministic: bool = False
    amp: bool = True
    cache: bool = False
    rect: bool = False
    val: bool = True
    plots: bool = True
    save_period: int = -1
    single_cls: bool = True

    optimizer: str = "auto"
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    close_mosaic: int = 10
    dropout: float = 0.0

    # Segmentation-specific Ultralytics options.
    overlap_mask: bool = True
    mask_ratio: int = 4

    # Training-time augmentation knobs passed to Ultralytics.
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    erasing: float = 0.4


@dataclass
class VisualizationConfig:
    """마스크 결과물 시각화 설정"""
    enabled: bool = True
    max_preview_size: int = 4096
    alpha: float = 0.45
    mask_color: List[int] = field(default_factory=lambda: [255, 0, 0])
    contour_color: List[int] = field(default_factory=lambda: [0, 255, 255])
    draw_contours: bool = True
    save_binary_mask: bool = False
    

@dataclass
class PipelineConfig:
    """전체 파이프라인 설정"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    postprocessing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # YAML 기반 하이퍼파라미터 override.
    # 예: hyperparameters.conf_threshold는 model.conf_threshold를 덮어쓴다.
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
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
        
        augmentation_dict = (
            config_dict.get('augmentation')
            or config_dict.get('aug')
            or {}
        )
        
        config = cls(
            data=DataConfig(**_filter_fields(DataConfig, data_dict)),
            model=ModelConfig(**_filter_fields(ModelConfig, config_dict.get('model', {}) or {})),
            gpu=GPUConfig(**_filter_fields(GPUConfig, config_dict.get('gpu', {}) or {})),
            postprocessing=PostProcessingConfig(
                **_filter_fields(PostProcessingConfig, config_dict.get('postprocessing', {}) or {})
            ),
            augmentation=AugmentationConfig(
                **_filter_fields(AugmentationConfig, augmentation_dict)
            ),
            training=TrainingConfig(
                **_filter_fields(
                    TrainingConfig,
                    config_dict.get('training') or config_dict.get('train') or {},
                )
            ),
            visualization=VisualizationConfig(
                **_filter_fields(VisualizationConfig, config_dict.get('visualization', {}) or {})
            ),
            hyperparameters=config_dict.get('hyperparameters', {}) or {},
            log_level=config_dict.get('log_level', "INFO"),
            save_intermediate=config_dict.get('save_intermediate', True),
        )
        config.apply_hyperparameters()
        return config

    def apply_hyperparameters(self) -> None:
        """hyperparameters 섹션을 실제 실행 설정에 반영"""
        hparams = self.hyperparameters or {}
        if not hparams:
            return
        
        section_map = {
            'data': self.data,
            'model': self.model,
            'gpu': self.gpu,
            'postprocessing': self.postprocessing,
            'augmentation': self.augmentation,
            'aug': self.augmentation,
            'training': self.training,
            'train': self.training,
            'visualization': self.visualization,
            'viz': self.visualization,
        }
        
        for section_name, section_config in section_map.items():
            values = hparams.get(section_name)
            if isinstance(values, dict):
                _assign_known_fields(section_config, values)
        
        flat_aliases = {
            'tile_size': (self.data, 'tile_size'),
            'tile_overlap_ratio': (self.data, 'overlap_ratio'),
            'overlap_ratio': (self.data, 'overlap_ratio'),
            'data_dir': (self.data, 'data_dir'),
            'output_dir': (self.data, 'output_dir'),
            'input_format': (self.data, 'input_format'),
            'model_name': (self.model, 'model_name'),
            'model_weights': (self.model, 'model_weights'),
            'conf_threshold': (self.model, 'conf_threshold'),
            'confidence_threshold': (self.model, 'conf_threshold'),
            'iou_threshold': (self.model, 'iou_threshold'),
            'sahi_slice_height': (self.model, 'sahi_slice_height'),
            'sahi_slice_width': (self.model, 'sahi_slice_width'),
            'sahi_overlap_height_ratio': (self.model, 'sahi_overlap_height_ratio'),
            'sahi_overlap_width_ratio': (self.model, 'sahi_overlap_width_ratio'),
            'mask_nms_threshold': (self.model, 'mask_nms_threshold'),
            'device_ids': (self.gpu, 'device_ids'),
            'device': (self.gpu, 'device'),
            'require_cuda': (self.gpu, 'require_cuda'),
            'allow_cpu_fallback': (self.gpu, 'allow_cpu_fallback'),
            'cuda_visible_devices': (self.gpu, 'cuda_visible_devices'),
            'use_multi_gpu': (self.gpu, 'use_multi_gpu'),
            'num_workers': (self.gpu, 'num_workers'),
            'batch_size': (self.gpu, 'batch_size'),
            'pin_memory': (self.gpu, 'pin_memory'),
            'simplification_tolerance': (self.postprocessing, 'simplification_tolerance'),
            'min_polygon_area': (self.postprocessing, 'min_polygon_area'),
            'enable_right_angle_regularization': (
                self.postprocessing,
                'enable_right_angle_regularization',
            ),
            'right_angle_tolerance': (self.postprocessing, 'right_angle_tolerance'),
            'output_formats': (self.postprocessing, 'output_formats'),
            'enable_tta': (self.augmentation, 'enable_tta'),
            'tta_scales': (self.augmentation, 'tta_scales'),
            'horizontal_flip': (self.augmentation, 'horizontal_flip'),
            'vertical_flip': (self.augmentation, 'vertical_flip'),
            'tta_merge_iou_threshold': (self.augmentation, 'merge_iou_threshold'),
            'merge_iou_threshold': (self.augmentation, 'merge_iou_threshold'),
            'save_mask_overlay': (self.visualization, 'enabled'),
            'visualization_enabled': (self.visualization, 'enabled'),
            'mask_preview_max_size': (self.visualization, 'max_preview_size'),
            'max_preview_size': (self.visualization, 'max_preview_size'),
            'mask_overlay_alpha': (self.visualization, 'alpha'),
            'mask_color': (self.visualization, 'mask_color'),
            'contour_color': (self.visualization, 'contour_color'),
            'draw_contours': (self.visualization, 'draw_contours'),
            'save_binary_mask': (self.visualization, 'save_binary_mask'),
            'training_enabled': (self.training, 'enabled'),
            'dataset_yaml': (self.training, 'dataset_yaml'),
            'train_images': (self.training, 'train_images'),
            'val_images': (self.training, 'val_images'),
            'train_labels': (self.training, 'train_labels'),
            'val_labels': (self.training, 'val_labels'),
            'use_train_as_val_if_missing': (self.training, 'use_train_as_val_if_missing'),
            'train_as_val': (self.training, 'use_train_as_val_if_missing'),
            'class_names': (self.training, 'class_names'),
            'pretrained_weights': (self.training, 'pretrained_weights'),
            'epochs': (self.training, 'epochs'),
            'training_epochs': (self.training, 'epochs'),
            'imgsz': (self.training, 'imgsz'),
            'train_imgsz': (self.training, 'imgsz'),
            'train_batch': (self.training, 'batch'),
            'train_workers': (self.training, 'workers'),
            'train_device': (self.training, 'device'),
            'train_project': (self.training, 'project'),
            'train_name': (self.training, 'name'),
            'learning_rate': (self.training, 'lr0'),
            'lr0': (self.training, 'lr0'),
            'lrf': (self.training, 'lrf'),
            'weight_decay': (self.training, 'weight_decay'),
            'warmup_epochs': (self.training, 'warmup_epochs'),
        }
        
        for key, value in hparams.items():
            if key in section_map:
                continue
            target = flat_aliases.get(key)
            if target is None:
                continue
            config_obj, attr_name = target
            if attr_name in {'data_dir', 'output_dir'}:
                value = Path(value)
            setattr(config_obj, attr_name, value)
    
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
            'augmentation': self.augmentation.__dict__,
            'training': self.training.__dict__,
            'visualization': self.visualization.__dict__,
            'hyperparameters': self.hyperparameters,
            'log_level': self.log_level,
            'save_intermediate': self.save_intermediate,
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


# 기본 설정 인스턴스
DEFAULT_CONFIG = PipelineConfig()


def _filter_fields(config_cls, values: Dict[str, Any]) -> Dict[str, Any]:
    """dataclass에 정의된 필드만 남긴다."""
    field_names = {field.name for field in fields(config_cls)}
    return {key: value for key, value in values.items() if key in field_names}


def _assign_known_fields(config_obj, values: Dict[str, Any]) -> None:
    """이미 생성된 dataclass 객체에 존재하는 필드만 할당한다."""
    field_names = {field.name for field in fields(config_obj)}
    for key, value in values.items():
        if key in field_names:
            if key in {'data_dir', 'output_dir'}:
                value = Path(value)
            setattr(config_obj, key, value)
