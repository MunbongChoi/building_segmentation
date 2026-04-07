"""전체 파이프라인 통합"""
import argparse
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.dataset import GeoTIFFLoader, GeoTile
from src.core.inference import SAHIInferenceEngine
from src.postprocessing.mask_to_polygon import MaskToPolygonConverter, RightAngleRegularizer
from src.postprocessing.geometry_utils import VectorFileWriter, GeometryUtils
from src.utils.config import PipelineConfig
from src.utils.logger import logger, setup_logger


class BuildingSegmentationPipeline:
    """건물 인스턴스 분할 완전 파이프라인"""
    
    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: 파이프라인 설정
        """
        self.config = config
        
        # 로깅 설정
        setup_logger(
            log_dir=str(config.data.output_dir / "logs"),
            log_level=config.log_level,
        )
        
        # 필수 컴포넌트 초기화
        self.data_loader = GeoTIFFLoader(
            tile_size=config.data.tile_size,
            overlap_ratio=config.data.overlap_ratio,
        )
        
        self.inference_engine = SAHIInferenceEngine(
            config.model,
            config.gpu,
            config.augmentation,
        )
        
        self.mask_converter = MaskToPolygonConverter(
            simplification_tolerance=config.postprocessing.simplification_tolerance,
            min_polygon_area=config.postprocessing.min_polygon_area,
        )
        
        self.right_angle_regularizer = RightAngleRegularizer(
            tolerance_degrees=config.postprocessing.right_angle_tolerance,
        )
        
        self.vector_writer = VectorFileWriter(
            crs="EPSG:4326",
            output_dir=str(config.data.output_dir),
        )
        
        logger.info(
            "Augmentation config: "
            f"enable_tta={config.augmentation.enable_tta}, "
            f"tta_scales={config.augmentation.tta_scales}, "
            f"horizontal_flip={config.augmentation.horizontal_flip}, "
            f"vertical_flip={config.augmentation.vertical_flip}"
        )
        logger.info("Pipeline initialized successfully")
    
    def process_geotiff(self, geotiff_path: str) -> Dict[str, Any]:
        """단일 GeoTIFF 파일 처리
        
        Args:
            geotiff_path: 입력 GeoTIFF 파일 경로
            
        Returns:
            처리 결과 (폴리곤, 통계 등)
        """
        logger.info(f"Processing: {geotiff_path}")
        
        # 1. GeoTIFF 메타데이터 로드. 실제 영상은 window 단위로 읽는다.
        geo_data = self.data_loader.load_metadata(geotiff_path)
        crs = geo_data['crs']
        transform = geo_data['transform']
        image_width = geo_data['width']
        image_height = geo_data['height']
        self.vector_writer.crs = str(crs) if crs else "EPSG:4326"
        
        logger.info(f"Image size: {image_width}x{image_height}, CRS: {self.vector_writer.crs}")
        
        # 2. Windowed tile 기반 추론
        all_masks = []
        all_boxes = []
        all_scores = []
        all_class_ids = []
        tile_count = 0
        devices = self.inference_engine.get_inference_devices()
        devices = devices[:max(1, min(len(devices), self.config.gpu.num_workers))]
        tile_batch = []
        
        for tile in self.data_loader.tile_geotiff(geotiff_path):
            tile_batch.append(tile)
            if len(tile_batch) < len(devices):
                continue
            
            tile_results = self._predict_tile_batch(
                tile_batch,
                devices,
                image_height=image_height,
                image_width=image_width,
            )
            tile_count += len(tile_batch)
            tile_batch = []
            
            for tile_result in tile_results:
                all_masks.extend(tile_result['masks'])
                if len(tile_result['boxes']) > 0:
                    all_boxes.append(tile_result['boxes'])
                    all_scores.append(tile_result['scores'])
                    all_class_ids.append(tile_result['class_ids'])
        
        if tile_batch:
            tile_results = self._predict_tile_batch(
                tile_batch,
                devices,
                image_height=image_height,
                image_width=image_width,
            )
            tile_count += len(tile_batch)
            
            for tile_result in tile_results:
                all_masks.extend(tile_result['masks'])
                if len(tile_result['boxes']) > 0:
                    all_boxes.append(tile_result['boxes'])
                    all_scores.append(tile_result['scores'])
                    all_class_ids.append(tile_result['class_ids'])
        
        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.empty((0, 4))
        scores = np.concatenate(all_scores, axis=0) if all_scores else np.empty(0)
        class_ids = np.concatenate(all_class_ids, axis=0) if all_class_ids else np.empty(0, dtype=int)
        
        # 3. 타일 간 중복 제거
        inference_result = self.inference_engine.nms_predictions(
            all_masks,
            boxes,
            scores,
            class_ids,
            image_height=image_height,
            image_width=image_width,
            iou_threshold=self.config.model.mask_nms_threshold,
        )
        
        masks = inference_result['masks']
        logger.info(f"Processed {tile_count} tiles; detected {len(masks)} buildings")
        
        # 4. 마스크 -> 픽셀 좌표계 폴리곤 변환
        polygons = self.mask_converter.masks_to_polygons(masks)
        logger.info(f"Converted to {len(polygons)} polygons")
        
        # 5. 직각화 (선택사항). 픽셀 단위 공차를 유지하기 위해 좌표 변환 전에 수행한다.
        if self.config.postprocessing.enable_right_angle_regularization:
            polygons = self.right_angle_regularizer.regularize_polygons(polygons)
            logger.info("Applied right-angle regularization")
        
        pixel_polygons = polygons
        geo_polygons = GeometryUtils.pixel_polygons_to_geo(pixel_polygons, transform)
        
        # 6. 메트릭 계산
        metrics_list = []
        for pixel_polygon, geo_polygon in zip(pixel_polygons, geo_polygons):
            metrics = GeometryUtils.calculate_building_metrics(geo_polygon)
            pixel_metrics = GeometryUtils.calculate_building_metrics(pixel_polygon)
            metrics['pixel_area'] = pixel_metrics['area']
            metrics['pixel_perimeter'] = pixel_metrics['perimeter']
            metrics_list.append(metrics)
        
        # 7. 벡터 파일 저장
        output_basename = Path(geotiff_path).stem
        output_files = {}
        
        if 'geojson' in self.config.postprocessing.output_formats:
            output_files['geojson'] = self.vector_writer.write_geojson(
                geo_polygons,
                output_basename,
                properties_list=metrics_list,
            )
        
        if 'shapefile' in self.config.postprocessing.output_formats:
            output_files['shapefile'] = self.vector_writer.write_shapefile(
                geo_polygons,
                output_basename,
                properties_list=metrics_list,
            )
        
        if 'json' in self.config.postprocessing.output_formats:
            output_files['json'] = self.vector_writer.write_json(
                geo_polygons,
                output_basename,
                properties_list=metrics_list,
            )
        
        # 8. 결과 요약
        result = {
            'input_file': geotiff_path,
            'num_buildings': len(geo_polygons),
            'polygons': geo_polygons,
            'pixel_polygons': pixel_polygons,
            'metrics': metrics_list,
            'output_files': output_files,
            'inference_result': inference_result,
            'crs': crs,
            'transform': transform,
        }
        
        logger.info(f"Processing complete: {len(polygons)} buildings detected and saved")
        
        return result

    def _predict_tile_batch(
        self,
        tiles: List[GeoTile],
        devices: List[str],
        image_height: int,
        image_width: int,
    ) -> List[Dict[str, Any]]:
        """타일 배치를 디바이스별로 1개씩 병렬 추론"""
        worker_count = min(len(tiles), len(devices))
        if worker_count <= 1:
            return [
                self._predict_single_tile(
                    tiles[0],
                    devices[0],
                    image_height=image_height,
                    image_width=image_width,
                )
            ]
        
        results = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    self._predict_single_tile,
                    tile,
                    devices[index],
                    image_height,
                    image_width,
                )
                for index, tile in enumerate(tiles)
            ]
            
            for future in as_completed(futures):
                results.append(future.result())
        
        return results

    def _predict_single_tile(
        self,
        tile: GeoTile,
        device: str,
        image_height: int,
        image_width: int,
    ) -> Dict[str, Any]:
        """단일 타일을 지정 디바이스에서 추론하고 전체 이미지 좌표로 이동"""
        logger.info(
            f"Tile {tile.tile_id} on {device}: offset=({tile.global_x}, {tile.global_y}), "
            f"size={tile.valid_width}x{tile.valid_height}"
        )
        
        tile_result = self.inference_engine.predict_with_slicing(
            tile.image,
            conf_threshold=self.config.model.conf_threshold,
            iou_threshold=self.config.model.iou_threshold,
            devices=[device],
        )
        return self.inference_engine.offset_result(
            tile_result,
            offset_x=tile.global_x,
            offset_y=tile.global_y,
            image_height=image_height,
            image_width=image_width,
        )
    
    def process_directory(self, input_dir: str) -> List[Dict[str, Any]]:
        """디렉토리 내 모든 GeoTIFF 처리
        
        Args:
            input_dir: 입력 디렉토리
            
        Returns:
            각 파일의 처리 결과 리스트
        """
        input_path = Path(input_dir)
        geotiff_files = list(input_path.glob("*.tif*")) + list(input_path.glob("*.geotiff"))
        
        logger.info(f"Found {len(geotiff_files)} GeoTIFF files in {input_dir}")
        
        results = []
        for i, geotiff_file in enumerate(geotiff_files, 1):
            try:
                logger.info(f"[{i}/{len(geotiff_files)}] Processing {geotiff_file.name}")
                result = self.process_geotiff(str(geotiff_file))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {geotiff_file}: {e}", exc_info=True)
        
        return results
    
    def cleanup(self):
        """리소스 정리"""
        self.inference_engine.clear_gpu_cache()
        logger.info("Pipeline cleanup complete")


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Building instance segmentation pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 설정 파일 경로. 미지정 시 configs/phase1_yolov8.yaml이 있으면 사용",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="처리할 GeoTIFF 파일 또는 디렉토리. 미지정 시 config.data.data_dir 사용",
    )
    return parser.parse_args()


def load_pipeline_config(config_path: str = None) -> PipelineConfig:
    """CLI 설정 로드"""
    if config_path:
        return PipelineConfig.from_yaml(config_path)
    
    default_config = Path("configs/phase1_yolov8.yaml")
    if default_config.exists():
        return PipelineConfig.from_yaml(str(default_config))
    
    return PipelineConfig()


def main():
    """메인 엔트리 포인트 (Phase 1)"""
    args = parse_args()
    config = load_pipeline_config(args.config)
    input_path = Path(args.input) if args.input else config.data.data_dir
    
    if not input_path.exists():
        setup_logger(
            log_dir=str(config.data.output_dir / "logs"),
            log_level=config.log_level,
        )
        logger.warning(f"Input path not found: {input_path}")
        logger.info("Creating sample config for reference...")
        config.to_yaml("configs/phase1_yolov8.yaml")
        return
    
    pipeline = BuildingSegmentationPipeline(config)
    try:
        if input_path.is_file():
            results = [pipeline.process_geotiff(str(input_path))]
        else:
            results = pipeline.process_directory(str(input_path))
        
        # 결과 요약
        total_buildings = sum(r['num_buildings'] for r in results)
        logger.info(f"\\n{'='*50}")
        logger.info(f"Pipeline complete!")
        logger.info(f"Files processed: {len(results)}")
        logger.info(f"Total buildings detected: {total_buildings}")
        logger.info(f"Output directory: {config.data.output_dir}")
        logger.info(f"{'='*50}")
        
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
