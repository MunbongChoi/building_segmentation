"""이미지 데이터 로딩 및 타일링"""
import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
from pathlib import Path
from typing import Tuple, Iterator, Dict, Any, List
from dataclasses import dataclass
import cv2

from ..utils.logger import logger


GEOTIFF_EXTENSIONS = {".tif", ".tiff", ".geotiff"}
RASTER_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SUPPORTED_IMAGE_EXTENSIONS = GEOTIFF_EXTENSIONS | RASTER_IMAGE_EXTENSIONS


@dataclass
class GeoTile:
    """이미지 타일. PNG/JPEG는 bounds/transform/crs가 None일 수 있다."""
    image: np.ndarray  # (H, W, C) 또는 (H, W)
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    transform: Any  # rasterio.transform.Affine
    crs: Any  # rasterio.crs.CRS
    tile_id: int
    global_x: int  # 전역 좌표계에서의 x 오프셋 (픽셀)
    global_y: int  # 전역 좌표계에서의 y 오프셋 (픽셀)
    valid_width: int  # 패딩을 제외한 실제 타일 폭
    valid_height: int  # 패딩을 제외한 실제 타일 높이


class GeoTIFFLoader:
    """GeoTIFF/PNG/JPEG 파일 로더"""
    
    def __init__(
        self,
        tile_size: int = 1024,
        overlap_ratio: float = 0.1,
        normalize: bool = True,
    ):
        """
        Args:
            tile_size: 타일 크기 (정사각형)
            overlap_ratio: 타일 간 오버랩 비율 (0.0 ~ 1.0)
            normalize: 이미지 정규화 여부 (0-255 -> 0-1)
        """
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.normalize = normalize
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not 0 <= overlap_ratio < 1:
            raise ValueError("overlap_ratio must be in the range [0, 1)")
        self.stride = max(1, int(tile_size * (1 - overlap_ratio)))
    
    def load_geotiff(self, file_path: str) -> Dict[str, Any]:
        """GeoTIFF 파일 로드
        
        Args:
            file_path: GeoTIFF 파일 경로
            
        Returns:
            {
                'image': np.ndarray (H, W, C),
                'width': int,
                'height': int,
                'bounds': (minx, miny, maxx, maxy),
                'transform': rasterio.transform.Affine,
                'crs': rasterio.crs.CRS,
                'metadata': dict,
            }
        """
        with rasterio.open(file_path) as src:
            image = src.read()  # (C, H, W)
            image = self._prepare_image(image)
            
            return {
                'image': image,
                'width': src.width,
                'height': src.height,
                'bounds': src.bounds,
                'transform': src.transform,
                'crs': src.crs,
                'metadata': src.meta,
            }

    def load_metadata(self, file_path: str) -> Dict[str, Any]:
        """이미지 배열을 읽지 않고 메타데이터만 로드"""
        if self._is_raster_image(file_path):
            return self.load_raster_metadata(file_path)
        
        with rasterio.open(file_path) as src:
            return {
                'width': src.width,
                'height': src.height,
                'bounds': src.bounds,
                'transform': src.transform,
                'crs': src.crs,
                'metadata': src.meta,
            }

    def load_raster_image(self, file_path: str) -> Dict[str, Any]:
        """PNG/JPEG 등 일반 raster image 로드"""
        image = self._read_raster_image(file_path)
        return {
            'image': self._prepare_raster_image(image),
            'width': image.shape[1],
            'height': image.shape[0],
            'bounds': None,
            'transform': None,
            'crs': None,
            'metadata': {'driver': Path(file_path).suffix.lower().lstrip('.')},
        }

    def load_raster_metadata(self, file_path: str) -> Dict[str, Any]:
        """PNG/JPEG 등 일반 raster image 메타데이터 로드"""
        image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {file_path}")
        
        return {
            'width': image.shape[1],
            'height': image.shape[0],
            'bounds': None,
            'transform': None,
            'crs': None,
            'metadata': {'driver': Path(file_path).suffix.lower().lstrip('.')},
        }
    
    def tile_geotiff(self, file_path: str) -> Iterator[GeoTile]:
        """GeoTIFF 파일을 타일로 분할
        
        Yields:
            GeoTile 인스턴스들
        """
        with rasterio.open(file_path) as src:
            x_offsets = self._axis_offsets(src.width)
            y_offsets = self._axis_offsets(src.height)
            tile_id = 0
            
            for y in y_offsets:
                for x in x_offsets:
                    window_width = min(self.tile_size, src.width - x)
                    window_height = min(self.tile_size, src.height - y)
                    window = Window(x, y, window_width, window_height)
                    
                    tile = self._prepare_image(src.read(window=window))
                    tile_transform = window_transform(window, src.transform)
                    tile_bounds = window_bounds(window, src.transform)
                    
                    geo_tile = GeoTile(
                        image=tile,
                        bounds=tile_bounds,
                        transform=tile_transform,
                        crs=src.crs,
                        tile_id=tile_id,
                        global_x=int(x),
                        global_y=int(y),
                        valid_width=int(window_width),
                        valid_height=int(window_height),
                    )
                    
                    tile_id += 1
                    yield geo_tile
                    
                    logger.debug(f"Yielded tile {tile_id-1}: ({x}, {y}) -> {tile_bounds}")

    def tile_image(self, file_path: str) -> Iterator[GeoTile]:
        """GeoTIFF 또는 일반 이미지 파일을 타일로 분할"""
        if not self._is_raster_image(file_path):
            yield from self.tile_geotiff(file_path)
            return
        
        image_data = self.load_raster_image(file_path)
        image = image_data['image']
        height, width = image.shape[:2]
        x_offsets = self._axis_offsets(width)
        y_offsets = self._axis_offsets(height)
        tile_id = 0
        
        for y in y_offsets:
            for x in x_offsets:
                window_width = min(self.tile_size, width - x)
                window_height = min(self.tile_size, height - y)
                tile = image[y:y + window_height, x:x + window_width].copy()
                
                yield GeoTile(
                    image=tile,
                    bounds=None,
                    transform=None,
                    crs=None,
                    tile_id=tile_id,
                    global_x=int(x),
                    global_y=int(y),
                    valid_width=int(window_width),
                    valid_height=int(window_height),
                )
                
                logger.debug(f"Yielded image tile {tile_id}: ({x}, {y})")
                tile_id += 1
    
    def get_tile_grid_info(self, file_path: str) -> Dict[str, int]:
        """타일 그리드 정보 반환"""
        metadata = self.load_metadata(file_path)
        width = metadata['width']
        height = metadata['height']
        
        num_x = len(self._axis_offsets(width))
        num_y = len(self._axis_offsets(height))
        total_tiles = num_x * num_y
        
        return {
            'num_x': num_x,
            'num_y': num_y,
            'total_tiles': total_tiles,
            'image_width': width,
            'image_height': height,
        }

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Rasterio CHW 배열을 모델 입력용 HWC 3채널 이미지로 변환"""
        # 필요시 정규화 (uint8 -> float32)
        if self.normalize and image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Rasterio read 결과는 CHW이므로 HWC로 변환
        if len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # 모델 입력은 RGB 3채널로 맞춘다.
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 2:
            image = np.concatenate([image, image[..., :1]], axis=-1)
        elif image.shape[-1] > 3:
            image = image[..., :3]
        
        return image

    def _read_raster_image(self, file_path: str) -> np.ndarray:
        """PNG/JPEG 등 일반 이미지 파일을 RGB/BGRA 기준 배열로 읽기"""
        image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {file_path}")
        
        if image.ndim == 2:
            return image
        if image.shape[-1] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[-1] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return image

    def _prepare_raster_image(self, image: np.ndarray) -> np.ndarray:
        """OpenCV로 읽은 일반 이미지 배열을 모델 입력용 HWC 3채널로 변환"""
        if self.normalize and image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 2:
            image = np.concatenate([image, image[..., :1]], axis=-1)
        elif image.shape[-1] > 3:
            image = image[..., :3]
        
        return image

    def _axis_offsets(self, length: int) -> List[int]:
        """축 하나에 대해 끝단을 보장하는 sliding-window 시작점 생성"""
        if length <= self.tile_size:
            return [0]
        
        offsets = list(range(0, length - self.tile_size + 1, self.stride))
        last_offset = length - self.tile_size
        if offsets[-1] != last_offset:
            offsets.append(last_offset)
        
        return offsets

    @staticmethod
    def _is_raster_image(file_path: str) -> bool:
        """PNG/JPEG 등 일반 raster image 여부"""
        return Path(file_path).suffix.lower() in RASTER_IMAGE_EXTENSIONS


class TileDataset:
    """타일 기반 데이터셋"""
    
    def __init__(self, geotiff_dir: str, tile_size: int = 1024, overlap_ratio: float = 0.1):
        self.geotiff_dir = Path(geotiff_dir)
        self.loader = GeoTIFFLoader(tile_size=tile_size, overlap_ratio=overlap_ratio)
        self.tile_files = [
            path
            for path in self.geotiff_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        
        logger.info(f"Found {len(self.tile_files)} GeoTIFF files in {geotiff_dir}")
    
    def __iter__(self) -> Iterator[GeoTile]:
        """타일 반복"""
        for file_path in self.tile_files:
            logger.info(f"Processing: {file_path}")
            yield from self.loader.tile_image(str(file_path))
