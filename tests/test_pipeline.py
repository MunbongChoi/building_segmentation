"""Phase 1 파이프라인 테스트"""
import numpy as np
import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
from affine import Affine

from src.core.dataset import GeoTIFFLoader
from src.core.inference import SAHIInferenceEngine
from src.postprocessing.mask_to_polygon import MaskToPolygonConverter, RightAngleRegularizer
from src.postprocessing.geometry_utils import GeometryUtils, VectorFileWriter
from src.utils.config import PipelineConfig


class TestDataLoader:
    """데이터 로더 테스트"""
    
    def test_loader_initialization(self):
        """로더 초기화 테스트"""
        loader = GeoTIFFLoader(tile_size=1024, overlap_ratio=0.1)
        assert loader.tile_size == 1024
        assert loader.overlap_ratio == 0.1
        assert loader.stride == 921  # 1024 * 0.9
    
    def test_tile_generation(self):
        """타일 생성 로직 테스트"""
        loader = GeoTIFFLoader(tile_size=256, overlap_ratio=0.1)
        slices = []
        
        # Mock 타일 생성
        image_h, image_w = 1000, 1000
        y = 0
        while y < image_h:
            x = 0
            while x < image_w:
                slices.append({'x': x, 'y': y})
                x += loader.stride
            y += loader.stride
        
        assert len(slices) > 0
        assert slices[0] == {'x': 0, 'y': 0}


class TestMaskToPolygon:
    """마스크→폴리곤 변환 테스트"""
    
    def test_converter_initialization(self):
        """컨버터 초기화 테스트"""
        converter = MaskToPolygonConverter(
            simplification_tolerance=1.0,
            min_polygon_area=100.0,
        )
        assert converter.simplification_tolerance == 1.0
        assert converter.min_polygon_area == 100.0
    
    def test_simple_mask_conversion(self):
        """간단한 마스크 변환 테스트"""
        converter = MaskToPolygonConverter()
        
        # 간단한 정사각형 마스크 생성
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255  # 60x60 정사각형
        
        polygon = converter.mask_to_polygon(mask)
        
        if polygon is not None:
            assert polygon.area > 0
            assert polygon.is_valid

    def test_offset_mask_record_conversion(self):
        """crop mask + offset 형식 변환 테스트"""
        converter = MaskToPolygonConverter()
        mask_record = {
            'mask': np.ones((10, 20), dtype=np.uint8),
            'offset_x': 100,
            'offset_y': 200,
        }
        
        polygon = converter.mask_to_polygon(mask_record)
        
        assert polygon is not None
        minx, miny, maxx, maxy = polygon.bounds
        assert minx >= 100
        assert miny >= 200
        assert maxx <= 120
        assert maxy <= 210


class TestRightAngleRegularization:
    """직각화 테스트"""
    
    def test_regularizer_initialization(self):
        """정규화기 초기화 테스트"""
        regularizer = RightAngleRegularizer(
            tolerance_degrees=5.0,
            snap_to_grid=False,
        )
        assert regularizer.tolerance_degrees == 5.0
    
    def test_nearest_right_angle(self):
        """가장 가까운 직각 찾기 테스트"""
        # 정확한 직각
        assert RightAngleRegularizer._find_nearest_right_angle(0) == 0
        assert RightAngleRegularizer._find_nearest_right_angle(90) == 90
        assert RightAngleRegularizer._find_nearest_right_angle(180) == 180
        assert RightAngleRegularizer._find_nearest_right_angle(270) == 270
        
        # 근처 각도
        assert RightAngleRegularizer._find_nearest_right_angle(88) == 90
        assert RightAngleRegularizer._find_nearest_right_angle(92) == 90


class TestGeometryUtils:
    """지오메트리 유틸리티 테스트"""
    
    def test_metrics_calculation(self):
        """건물 메트릭 계산 테스트"""
        from shapely.geometry import Polygon
        
        # 정사각형 폴리곤 (100x100)
        polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        
        metrics = GeometryUtils.calculate_building_metrics(polygon)
        
        assert 'area' in metrics
        assert 'perimeter' in metrics
        assert 'width' in metrics
        assert 'height' in metrics
        assert abs(metrics['area'] - 10000) < 1  # 100*100
        assert abs(metrics['perimeter'] - 400) < 1  # 4*100

    def test_pixel_polygon_to_geo(self):
        """픽셀 좌표를 GeoTIFF 좌표로 변환"""
        from shapely.geometry import Polygon
        
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        transform = Affine.translation(1000, 2000) * Affine.scale(0.5, -0.5)
        
        geo_polygon = GeometryUtils.pixel_polygon_to_geo(polygon, transform)
        
        assert geo_polygon.bounds == pytest.approx((1000, 1995, 1005, 2000))


class TestInferenceMasks:
    """추론 마스크 병합 테스트"""
    
    def test_crop_mask_iou(self):
        """전체 마스크 생성 없이 crop mask record 간 IOU 계산"""
        mask1 = {
            'mask': np.ones((10, 10), dtype=np.uint8),
            'offset_x': 0,
            'offset_y': 0,
        }
        mask2 = {
            'mask': np.ones((10, 10), dtype=np.uint8),
            'offset_x': 5,
            'offset_y': 0,
        }
        
        iou = SAHIInferenceEngine._mask_iou(mask1, mask2)
        
        assert iou == pytest.approx(5 * 10 / (100 + 100 - 50))


class TestPipelineConfig:
    """파이프라인 설정 테스트"""
    
    def test_config_initialization(self):
        """설정 초기화 테스트"""
        config = PipelineConfig()
        
        assert config.data.tile_size == 1024
        assert config.model.conf_threshold == 0.4
        assert len(config.gpu.device_ids) == 4
        assert config.gpu.device == "cuda"
        assert config.gpu.require_cuda is True
        assert config.gpu.allow_cpu_fallback is False
        assert config.postprocessing.simplification_tolerance == 1.0
        assert config.augmentation.enable_tta is False
        assert config.visualization.enabled is True
    
    def test_config_yaml_export(self, tmp_path):
        """YAML 내보내기 테스트"""
        config = PipelineConfig()
        yaml_path = tmp_path / "test_config.yaml"
        
        config.to_yaml(str(yaml_path))
        
        assert yaml_path.exists()
        
        # 로드 테스트
        loaded_config = PipelineConfig.from_yaml(str(yaml_path))
        assert loaded_config.data.tile_size == config.data.tile_size
        assert isinstance(loaded_config.data.data_dir, Path)

    def test_config_hyperparameter_override(self, tmp_path):
        """YAML hyperparameters override 테스트"""
        yaml_path = tmp_path / "hparams.yaml"
        yaml_path.write_text(
            """
data:
  data_dir: ./data
model:
  conf_threshold: 0.2
hyperparameters:
  conf_threshold: 0.7
  tile_size: 512
  device: cpu
  require_cuda: false
  allow_cpu_fallback: true
  enable_tta: true
  tta_scales: [1.0, 1.25]
  save_mask_overlay: false
augmentation:
  enable_tta: false
visualization:
  enabled: true
""",
            encoding="utf-8",
        )
        
        loaded_config = PipelineConfig.from_yaml(str(yaml_path))
        
        assert loaded_config.model.conf_threshold == 0.7
        assert loaded_config.data.tile_size == 512
        assert loaded_config.gpu.device == "cpu"
        assert loaded_config.gpu.require_cuda is False
        assert loaded_config.gpu.allow_cpu_fallback is True
        assert loaded_config.augmentation.enable_tta is True
        assert loaded_config.augmentation.tta_scales == [1.0, 1.25]
        assert loaded_config.visualization.enabled is False


class TestYOLOTrainingConfig:
    def test_missing_validation_paths_can_reuse_training_data(self, tmp_path):
        from src.core.training import YOLOSegFineTuner

        train_images = tmp_path / "images" / "train"
        train_labels = tmp_path / "labels" / "train"
        train_images.mkdir(parents=True)
        train_labels.mkdir(parents=True)
        (train_images / "sample.png").write_bytes(b"")
        (train_labels / "sample.txt").write_text("0 0 0 1 0 1 1\n", encoding="utf-8")

        config = PipelineConfig()
        config.data.output_dir = tmp_path / "outputs"
        config.training.train_images = str(train_images)
        config.training.train_labels = str(train_labels)
        config.training.val_images = None
        config.training.val_labels = None
        config.training.use_train_as_val_if_missing = True

        dataset_yaml = YOLOSegFineTuner(config).prepare_dataset_yaml()

        dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        assert dataset["train"] == train_images.resolve().as_posix()
        assert dataset["val"] == train_images.resolve().as_posix()

    def test_missing_validation_paths_fail_when_train_as_val_is_disabled(self, tmp_path):
        from src.core.training import YOLOSegFineTuner

        train_images = tmp_path / "images" / "train"
        train_labels = tmp_path / "labels" / "train"
        train_images.mkdir(parents=True)
        train_labels.mkdir(parents=True)
        (train_images / "sample.png").write_bytes(b"")
        (train_labels / "sample.txt").write_text("0 0 0 1 0 1 1\n", encoding="utf-8")

        config = PipelineConfig()
        config.data.output_dir = tmp_path / "outputs"
        config.training.train_images = str(train_images)
        config.training.train_labels = str(train_labels)
        config.training.val_images = None
        config.training.val_labels = None
        config.training.use_train_as_val_if_missing = False

        with pytest.raises(FileNotFoundError, match="training.val_images is not configured"):
            YOLOSegFineTuner(config).prepare_dataset_yaml()


class TestIntegration:
    """통합 테스트"""
    
    def test_pipeline_initialization(self):
        """파이프라인 초기화 테스트"""
        from main import BuildingSegmentationPipeline
        
        config = PipelineConfig()
        with patch("main.SAHIInferenceEngine"):
            pipeline = BuildingSegmentationPipeline(config)
        
        assert pipeline.config is not None
        assert pipeline.data_loader is not None
        assert pipeline.mask_converter is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
