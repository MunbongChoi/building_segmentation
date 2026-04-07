"""간단한 예제: Phase 1 파이프라인 사용법"""
import numpy as np
from pathlib import Path
from src.utils.config import PipelineConfig
from src.core.dataset import GeoTIFFLoader
from src.core.model_manager import MultiGPUModelManager
from src.core.inference import SAHIInferenceEngine
from src.postprocessing.mask_to_polygon import MaskToPolygonConverter, RightAngleRegularizer
from src.postprocessing.geometry_utils import VectorFileWriter, GeometryUtils
from src.utils.logger import setup_logger, get_logger


def example_full_pipeline():
    """완전한 파이프라인 예제"""
    
    # 로깅 설정
    setup_logger(log_level="INFO")
    logger = get_logger(__name__)
    
    # 1. 설정 로드
    logger.info("1. Loading configuration...")
    config = PipelineConfig()
    
    # 2. 모델 초기화
    logger.info("2. Initializing model...")
    inference_engine = SAHIInferenceEngine(config.model, config.gpu)
    
    # 3. 후처리 도구 초기화
    logger.info("3. Initializing post-processing tools...")
    mask_converter = MaskToPolygonConverter(
        simplification_tolerance=config.postprocessing.simplification_tolerance,
        min_polygon_area=config.postprocessing.min_polygon_area,
    )
    
    regularizer = RightAngleRegularizer(
        tolerance_degrees=config.postprocessing.right_angle_tolerance,
    )
    
    vector_writer = VectorFileWriter(output_dir=str(config.data.output_dir))
    
    # 4. 샘플 입력 이미지 생성 (실제로는 GeoTIFF 로드)
    logger.info("4. Creating sample image...")
    sample_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    
    # 5. SAHI 추론
    logger.info("5. Running SAHI inference...")
    inference_result = inference_engine.predict_with_slicing(
        sample_image,
        conf_threshold=config.model.conf_threshold,
        iou_threshold=config.model.iou_threshold,
    )
    
    masks = inference_result['masks']
    logger.info(f"Detected {len(masks)} objects")
    
    # 6. 마스크 -> 폴리곤 변환
    logger.info("6. Converting masks to polygons...")
    polygons = mask_converter.masks_to_polygons(masks)
    logger.info(f"Converted to {len(polygons)} polygons")
    
    # 7. 직각화 (선택)
    logger.info("7. Applying right-angle regularization...")
    polygons = regularizer.regularize_polygons(polygons)
    
    # 8. 메트릭 계산
    logger.info("8. Calculating metrics...")
    metrics_list = [GeometryUtils.calculate_building_metrics(p) for p in polygons]
    
    # 9. 벡터 파일 저장
    logger.info("9. Saving vector files...")
    
    # GeoJSON
    geojson_path = vector_writer.write_geojson(
        polygons,
        "sample_buildings",
        properties_list=metrics_list,
    )
    logger.info(f"GeoJSON saved: {geojson_path}")
    
    # Shapefile
    shp_path = vector_writer.write_shapefile(
        polygons,
        "sample_buildings",
        properties_list=metrics_list,
    )
    logger.info(f"Shapefile saved: {shp_path}")
    
    # JSON
    json_path = vector_writer.write_json(
        polygons,
        "sample_buildings",
        properties_list=metrics_list,
    )
    logger.info(f"JSON saved: {json_path}")
    
    logger.info("Pipeline complete!")
    
    return {
        'polygons': polygons,
        'metrics': metrics_list,
        'output_files': {
            'geojson': geojson_path,
            'shapefile': shp_path,
            'json': json_path,
        }
    }


def example_data_loader():
    """데이터 로더 예제"""
    logger = get_logger(__name__)
    
    logger.info("Data Loader Example")
    
    # GeoTIFF 로더 생성
    loader = GeoTIFFLoader(tile_size=1024, overlap_ratio=0.1)
    logger.info("GeoTIFFLoader created")
    
    # 타일 그리드 정보 조회 (샘플)
    # grid_info = loader.get_tile_grid_info("path/to/geotiff.tif")
    # logger.info(f"Grid info: {grid_info}")


def example_multi_gpu():
    """Multi-GPU 설정 예제"""
    logger = get_logger(__name__)
    
    import torch
    
    logger.info("Multi-GPU Example")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")


def example_config_management():
    """설정 관리 예제"""
    logger = get_logger(__name__)
    
    logger.info("Configuration Management Example")
    
    # 기본 설정 생성
    config = PipelineConfig()
    logger.info(f"Default config: tile_size={config.data.tile_size}")
    
    # 설정 수정
    config.model.conf_threshold = 0.5
    config.data.tile_size = 512
    logger.info(f"Modified config: conf_threshold={config.model.conf_threshold}")
    
    # YAML 저장
    yaml_path = "configs/custom_config.yaml"
    config.to_yaml(yaml_path)
    logger.info(f"Config saved to {yaml_path}")
    
    # YAML 로드
    loaded_config = PipelineConfig.from_yaml(yaml_path)
    logger.info(f"Loaded config: tile_size={loaded_config.data.tile_size}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        
        if example_name == "full_pipeline":
            example_full_pipeline()
        elif example_name == "data_loader":
            example_data_loader()
        elif example_name == "multi_gpu":
            example_multi_gpu()
        elif example_name == "config":
            example_config_management()
        else:
            print("Unknown example. Available examples:")
            print("  - full_pipeline: 완전한 파이프라인")
            print("  - data_loader: 데이터 로더")
            print("  - multi_gpu: Multi-GPU 설정")
            print("  - config: 설정 관리")
    else:
        print("Usage: python examples.py <example_name>")
        print("\nAvailable examples:")
        print("  - full_pipeline: 완전한 파이프라인")
        print("  - data_loader: 데이터 로더")
        print("  - multi_gpu: Multi-GPU 설정")
        print("  - config: 설정 관리")
