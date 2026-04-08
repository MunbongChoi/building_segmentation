"""Phase 1 파이프라인 사용 예제.

예제는 실제 구현 경로를 호출한다. 즉, full_pipeline은 PNG/JPEG/GeoTIFF를 tile 단위로
읽고, 타일/GPU 병렬 추론, 마스크 병합, 벡터/마스크 overlay 저장까지 수행한다.
GeoTIFF는 GIS 좌표로, PNG/JPEG는 픽셀 좌표로 저장한다.
"""
import argparse
from pathlib import Path
from typing import Optional


def _load_config(config_path: Optional[str] = None):
    """설정 파일이 있으면 YAML에서, 없으면 기본 설정으로 로드."""
    from src.utils.config import PipelineConfig

    if config_path:
        path = Path(config_path)
        if path.exists():
            return PipelineConfig.from_yaml(str(path))
        raise FileNotFoundError(f"Config file not found: {path}")

    default_path = Path("configs/phase1_yolov8.yaml")
    if default_path.exists():
        return PipelineConfig.from_yaml(str(default_path))

    return PipelineConfig()


def example_full_pipeline(input_path: Optional[str] = None, config_path: Optional[str] = None):
    """완전한 GeoTIFF 파이프라인 예제."""
    from main import BuildingSegmentationPipeline

    config = _load_config(config_path)
    target_path = Path(input_path) if input_path else config.data.data_dir

    if not target_path.exists():
        raise FileNotFoundError(
            f"Input path not found: {target_path}. "
            "PNG 파일들을 data/에 넣거나 --input 경로를 지정하세요."
        )

    pipeline = BuildingSegmentationPipeline(config)
    try:
        if target_path.is_file():
            return pipeline.process_geotiff(str(target_path))
        return pipeline.process_directory(str(target_path))
    finally:
        pipeline.cleanup()


def example_data_loader(input_path: Optional[str] = None, config_path: Optional[str] = None):
    """PNG/JPEG/GeoTIFF 데이터 로더 예제."""
    from src.core.dataset import GeoTIFFLoader
    from src.utils.logger import get_logger, setup_logger

    setup_logger(log_level="INFO")
    logger = get_logger(__name__)
    config = _load_config(config_path)
    loader = GeoTIFFLoader(
        tile_size=config.data.tile_size,
        overlap_ratio=config.data.overlap_ratio,
    )

    logger.info(
        f"GeoTIFFLoader: tile_size={loader.tile_size}, "
        f"overlap_ratio={loader.overlap_ratio}, stride={loader.stride}"
    )

    if not input_path:
        logger.info("Use --input <image_path> to inspect metadata and tile windows.")
        return None

    path = Path(input_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    metadata = loader.load_metadata(str(path))
    grid_info = loader.get_tile_grid_info(str(path))
    logger.info(
        f"Metadata: width={metadata['width']}, height={metadata['height']}, "
        f"crs={metadata['crs']}"
    )
    logger.info(f"Grid info: {grid_info}")

    for tile_index, tile in enumerate(loader.tile_image(str(path))):
        logger.info(
            f"Tile {tile.tile_id}: offset=({tile.global_x}, {tile.global_y}), "
            f"valid_size={tile.valid_width}x{tile.valid_height}, bounds={tile.bounds}"
        )
        if tile_index >= 2:
            logger.info("Only the first 3 tiles are displayed.")
            break

    return grid_info


def example_multi_gpu(config_path: Optional[str] = None, load_model: bool = False):
    """Multi-GPU 설정 및 실제 추론 디바이스 예제."""
    from src.utils.logger import get_logger, setup_logger

    setup_logger(log_level="INFO")
    logger = get_logger(__name__)
    config = _load_config(config_path)

    try:
        import torch
    except ImportError:
        logger.warning("PyTorch is not installed in this environment.")
        return []

    logger.info(f"Configured device_ids: {config.gpu.device_ids}")
    logger.info(f"device: {config.gpu.device}")
    logger.info(f"require_cuda: {config.gpu.require_cuda}")
    logger.info(f"allow_cpu_fallback: {config.gpu.allow_cpu_fallback}")
    logger.info(f"cuda_visible_devices: {config.gpu.cuda_visible_devices or '<env>'}")
    logger.info(f"use_multi_gpu: {config.gpu.use_multi_gpu}")
    logger.info(f"num_workers: {config.gpu.num_workers}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")

    for device_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device_id)
        logger.info(
            f"GPU {device_id}: {props.name}, "
            f"memory={props.total_memory / 1024**3:.1f}GB"
        )

    if not load_model:
        logger.info("Use --load-model to instantiate one YOLO model per inference device.")
        return []

    from src.core.model_manager import MultiGPUModelManager

    manager = MultiGPUModelManager(config.model, config.gpu)
    devices = manager.get_inference_devices()
    logger.info(f"Inference devices: {devices}")
    manager.clear_gpu_cache()
    return devices


def example_finetune(config_path: Optional[str] = None):
    """YOLOv8-Seg fine-tuning example."""
    from src.core.training import YOLOSegFineTuner
    from src.utils.logger import setup_logger

    config = _load_config(config_path)
    config.training.enabled = True
    setup_logger(
        log_dir=str(config.data.output_dir / "logs"),
        log_level=config.log_level,
        name="training",
    )
    trainer = YOLOSegFineTuner(config)
    return trainer.train()


def example_config_management(output_path: str = "configs/custom_config.yaml"):
    """설정 생성, 저장, 로드 예제."""
    from src.utils.config import PipelineConfig
    from src.utils.logger import get_logger, setup_logger

    setup_logger(log_level="INFO")
    logger = get_logger(__name__)

    config = PipelineConfig()
    logger.info(f"Default tile_size={config.data.tile_size}")

    config.model.conf_threshold = 0.5
    config.data.tile_size = 1024
    config.gpu.num_workers = min(4, len(config.gpu.device_ids))
    config.augmentation.enable_tta = True
    config.augmentation.tta_scales = [1.0, 1.25]
    config.augmentation.horizontal_flip = True
    config.visualization.enabled = True
    config.visualization.max_preview_size = 4096
    config.hyperparameters = {
        'conf_threshold': config.model.conf_threshold,
        'tile_size': config.data.tile_size,
        'num_workers': config.gpu.num_workers,
        'enable_tta': config.augmentation.enable_tta,
        'tta_scales': config.augmentation.tta_scales,
        'horizontal_flip': config.augmentation.horizontal_flip,
        'save_mask_overlay': config.visualization.enabled,
        'mask_preview_max_size': config.visualization.max_preview_size,
    }

    config.to_yaml(output_path)
    logger.info(f"Config saved to {output_path}")

    loaded_config = PipelineConfig.from_yaml(output_path)
    logger.info(
        f"Loaded config: tile_size={loaded_config.data.tile_size}, "
        f"num_workers={loaded_config.gpu.num_workers}, "
        f"enable_tta={loaded_config.augmentation.enable_tta}, "
        f"save_mask_overlay={loaded_config.visualization.enabled}"
    )
    return loaded_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Building instance segmentation examples"
    )
    subparsers = parser.add_subparsers(dest="example_name")

    full_pipeline = subparsers.add_parser("full_pipeline", help="완전한 파이프라인")
    full_pipeline.add_argument("--input", type=str, help="PNG/JPEG/GeoTIFF 파일 또는 디렉토리")
    full_pipeline.add_argument("--config", type=str, help="YAML 설정 파일")

    data_loader = subparsers.add_parser("data_loader", help="데이터 로더")
    data_loader.add_argument("--input", type=str, help="PNG/JPEG/GeoTIFF 파일")
    data_loader.add_argument("--config", type=str, help="YAML 설정 파일")

    multi_gpu = subparsers.add_parser("multi_gpu", help="Multi-GPU 설정")
    multi_gpu.add_argument("--config", type=str, help="YAML 설정 파일")
    multi_gpu.add_argument(
        "--load-model",
        action="store_true",
        help="GPU별 모델 인스턴스 로드까지 수행",
    )

    finetune = subparsers.add_parser("finetune", help="YOLOv8-Seg fine-tuning")
    finetune.add_argument("--config", type=str, help="YAML config file")

    config = subparsers.add_parser("config", help="설정 관리")
    config.add_argument(
        "--output",
        type=str,
        default="configs/custom_config.yaml",
        help="저장할 YAML 설정 파일",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.example_name == "full_pipeline":
        example_full_pipeline(input_path=args.input, config_path=args.config)
    elif args.example_name == "data_loader":
        example_data_loader(input_path=args.input, config_path=args.config)
    elif args.example_name == "multi_gpu":
        example_multi_gpu(config_path=args.config, load_model=args.load_model)
    elif args.example_name == "finetune":
        example_finetune(config_path=args.config)
    elif args.example_name == "config":
        example_config_management(output_path=args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
