"""유틸리티 스크립트"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

from src.utils.logger import setup_logger, get_logger

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig


def setup_directories(config: "PipelineConfig") -> None:
    """필요한 디렉토리 생성"""
    logger = get_logger(__name__)
    
    dirs = [
        config.data.data_dir,
        config.data.output_dir,
        config.data.output_dir / "logs",
        Path("configs"),
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {dir_path}")


def create_sample_config(output_path: str = "configs/phase1_yolov8.yaml") -> None:
    """샘플 설정 파일 생성"""
    logger = get_logger(__name__)
    from src.utils.config import PipelineConfig
    
    config = PipelineConfig()
    config.to_yaml(output_path)
    
    logger.info(f"Sample config created: {output_path}")


def download_model_weights(model_name: str = "yolov8m-seg") -> None:
    """YOLOv8 모델 가중치 다운로드"""
    logger = get_logger(__name__)
    
    try:
        from ultralytics import YOLO
        
        logger.info(f"Downloading {model_name}...")
        model = YOLO(f"{model_name}.pt")
        logger.info(f"Model downloaded successfully: {model_name}.pt")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")


def validate_environment() -> None:
    """환경 검증"""
    logger = get_logger(__name__)
    
    logger.info("=== Environment Validation ===")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    
    # PyTorch
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"PyTorch CUDA build: {torch.version.cuda}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  - GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
        else:
            logger.error(
                "CUDA is not available to PyTorch. Install a CUDA-enabled torch build "
                "and check the NVIDIA driver/container runtime."
            )
    except ImportError:
        logger.error("PyTorch is not installed.")
    
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            logger.info("nvidia-smi GPUs:")
            for line in result.stdout.strip().splitlines():
                logger.info(f"  {line}")
        else:
            logger.error(f"nvidia-smi failed: {result.stderr.strip()}")
    else:
        logger.error("nvidia-smi not found in PATH.")
    
    # Rasterio
    try:
        import rasterio
        logger.info(f"Rasterio: {rasterio.__version__}")
    except ImportError:
        logger.error("Rasterio is not installed.")
    
    # OpenCV
    try:
        import cv2
        logger.info(f"OpenCV: {cv2.__version__}")
    except ImportError:
        logger.error("OpenCV is not installed.")
    
    # NumPy
    try:
        import numpy as np
        logger.info(f"NumPy: {np.__version__}")
    except ImportError:
        logger.error("NumPy is not installed.")
    
    logger.info("=== Environment Validation Complete ===")


def cleanup_outputs(output_dir: str = "./outputs") -> None:
    """출력 디렉토리 정리"""
    logger = get_logger(__name__)
    
    output_path = Path(output_dir)
    
    if output_path.exists():
        response = input(f"Delete all files in {output_dir}? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleaned up: {output_dir}")
        else:
            logger.info("Cleanup cancelled")
    else:
        logger.info(f"Directory not found: {output_dir}")


def main():
    """메인 유틸리티"""
    parser = argparse.ArgumentParser(
        description="Building Segmentation Pipeline Utilities"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # setup 커맨드
    subparsers.add_parser("setup", help="Setup directories and create sample config")
    
    # validate 커맨드
    subparsers.add_parser("validate", help="Validate environment")
    
    # download 커맨드
    download_parser = subparsers.add_parser("download", help="Download model weights")
    download_parser.add_argument(
        "--model",
        type=str,
        default="yolov8m-seg",
        help="Model name (default: yolov8m-seg)"
    )
    
    # cleanup 커맨드
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up outputs")
    cleanup_parser.add_argument(
        "--dir",
        type=str,
        default="./outputs",
        help="Output directory (default: ./outputs)"
    )

    convert_parser = subparsers.add_parser("tif-to-png", help="Convert TIF/GeoTIFF files to PNG")
    convert_parser.add_argument("--input", required=True, help="TIF file or directory")
    convert_parser.add_argument("--output-dir", default="data", help="PNG output directory")
    convert_parser.add_argument(
        "--bands",
        type=int,
        nargs="+",
        default=None,
        help="One grayscale band or three RGB/false-color bands. Example: --bands 4 3 2",
    )
    convert_parser.add_argument(
        "--normalize",
        choices=["auto", "none", "minmax", "percentile"],
        default="auto",
        help="8-bit scaling method",
    )
    convert_parser.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        default=(2.0, 98.0),
        metavar=("LOW", "HIGH"),
        help="Percentile stretch range used by --normalize percentile/auto",
    )
    convert_parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="If > 0, split each TIF into tiled PNGs of this size",
    )
    convert_parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.0,
        help="Tile overlap ratio when --tile-size is used",
    )
    convert_parser.add_argument("--recursive", action="store_true", help="Search input directories recursively")
    convert_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files")
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logger(log_level="INFO")
    logger = get_logger(__name__)
    
    if args.command == "setup":
        logger.info("Running setup...")
        from src.utils.config import PipelineConfig
        
        config = PipelineConfig()
        setup_directories(config)
        create_sample_config()
        logger.info("Setup complete!")
    
    elif args.command == "validate":
        validate_environment()
    
    elif args.command == "download":
        download_model_weights(args.model)
    
    elif args.command == "cleanup":
        cleanup_outputs(args.dir)

    elif args.command == "tif-to-png":
        from src.utils.tif_to_png import convert_tif_to_png

        convert_tif_to_png(
            input_path=args.input,
            output_dir=args.output_dir,
            bands=args.bands,
            normalize=args.normalize,
            percentiles=tuple(args.percentiles),
            tile_size=args.tile_size,
            overlap_ratio=args.overlap_ratio,
            recursive=args.recursive,
            overwrite=args.overwrite,
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
