"""마스크 결과 시각화 유틸리티"""
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling

from ..core.dataset import RASTER_IMAGE_EXTENSIONS
from ..utils.config import VisualizationConfig
from ..utils.logger import logger


class MaskVisualizer:
    """원본 이미지 위에 인스턴스 마스크를 축소 오버레이로 저장"""
    
    def __init__(
        self,
        config: VisualizationConfig,
        output_dir: str = "./outputs",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_mask_preview(
        self,
        geotiff_path: str,
        masks: List[Any],
        filename: str,
    ) -> Dict[str, str]:
        """마스크 오버레이 preview 이미지를 저장"""
        if not self.config.enabled:
            return {}
        
        preview, scale_x, scale_y = self._load_preview(geotiff_path)
        mask_layer = self._rasterize_masks(
            masks,
            preview_height=preview.shape[0],
            preview_width=preview.shape[1],
            scale_x=scale_x,
            scale_y=scale_y,
        )
        
        overlay = self._blend_overlay(preview, mask_layer)
        output_files = {}
        
        overlay_path = self.output_dir / f"{filename}_mask_overlay.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        output_files['mask_overlay'] = str(overlay_path)
        logger.info(f"Mask overlay saved: {overlay_path}")
        
        if self.config.save_binary_mask:
            binary_path = self.output_dir / f"{filename}_mask_binary.png"
            cv2.imwrite(str(binary_path), mask_layer)
            output_files['mask_binary'] = str(binary_path)
            logger.info(f"Mask binary preview saved: {binary_path}")
        
        return output_files
    
    def _load_preview(self, geotiff_path: str) -> Tuple[np.ndarray, float, float]:
        """이미지를 preview 크기로 축소해서 RGB uint8로 읽는다."""
        if Path(geotiff_path).suffix.lower() in RASTER_IMAGE_EXTENSIONS:
            return self._load_raster_preview(geotiff_path)
        
        with rasterio.open(geotiff_path) as src:
            original_width = src.width
            original_height = src.height
            max_size = max(1, int(self.config.max_preview_size))
            ratio = min(1.0, max_size / max(original_width, original_height))
            preview_width = max(1, int(round(original_width * ratio)))
            preview_height = max(1, int(round(original_height * ratio)))
            
            image = src.read(
                out_shape=(src.count, preview_height, preview_width),
                resampling=Resampling.bilinear,
            )
        
        image = self._chw_to_rgb_uint8(image)
        scale_x = preview_width / original_width
        scale_y = preview_height / original_height
        return image, scale_x, scale_y

    def _load_raster_preview(self, image_path: str) -> Tuple[np.ndarray, float, float]:
        """PNG/JPEG 등 일반 이미지를 preview 크기로 로드"""
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[-1] > 3:
            image = image[..., :3]
        
        image = self._chw_to_rgb_uint8(image)
        original_height, original_width = image.shape[:2]
        max_size = max(1, int(self.config.max_preview_size))
        ratio = min(1.0, max_size / max(original_width, original_height))
        preview_width = max(1, int(round(original_width * ratio)))
        preview_height = max(1, int(round(original_height * ratio)))
        
        if ratio < 1.0:
            image = cv2.resize(
                image,
                (preview_width, preview_height),
                interpolation=cv2.INTER_AREA,
            )
        
        return image, preview_width / original_width, preview_height / original_height
    
    @staticmethod
    def _chw_to_rgb_uint8(image: np.ndarray) -> np.ndarray:
        """Rasterio CHW 배열을 RGB uint8 HWC 배열로 변환"""
        if image.ndim == 3 and image.shape[0] <= 4 and image.shape[-1] > 4:
            image = np.transpose(image, (1, 2, 0))
        
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 2:
            image = np.concatenate([image, image[..., :1]], axis=-1)
        elif image.shape[-1] > 3:
            image = image[..., :3]
        
        if image.dtype == np.uint8:
            return image.copy()
        
        image = image.astype(np.float32)
        output = np.zeros_like(image, dtype=np.uint8)
        for channel in range(image.shape[-1]):
            band = image[..., channel]
            finite = band[np.isfinite(band)]
            if finite.size == 0:
                continue
            low, high = np.percentile(finite, [2, 98])
            if high <= low:
                low = float(finite.min())
                high = float(finite.max())
            if high <= low:
                continue
            scaled = (band - low) / (high - low)
            output[..., channel] = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
        
        return output
    
    def _rasterize_masks(
        self,
        masks: List[Any],
        preview_height: int,
        preview_width: int,
        scale_x: float,
        scale_y: float,
    ) -> np.ndarray:
        """전역 픽셀 좌표 마스크를 preview 좌표계 mask layer로 변환"""
        mask_layer = np.zeros((preview_height, preview_width), dtype=np.uint8)
        
        for mask in masks:
            record = self._normalize_mask_record(mask)
            if record is None:
                continue
            
            mask_array = record['mask']
            x0 = int(np.floor(record['offset_x'] * scale_x))
            y0 = int(np.floor(record['offset_y'] * scale_y))
            x1 = int(np.ceil((record['offset_x'] + record['width']) * scale_x))
            y1 = int(np.ceil((record['offset_y'] + record['height']) * scale_y))
            
            x0 = max(0, min(preview_width, x0))
            y0 = max(0, min(preview_height, y0))
            x1 = max(0, min(preview_width, x1))
            y1 = max(0, min(preview_height, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            
            resized_mask = cv2.resize(
                mask_array,
                (x1 - x0, y1 - y0),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_layer[y0:y1, x0:x1] = np.maximum(
                mask_layer[y0:y1, x0:x1],
                (resized_mask > 0).astype(np.uint8) * 255,
            )
        
        return mask_layer
    
    @staticmethod
    def _normalize_mask_record(mask: Any) -> Dict[str, Any]:
        """crop mask record 또는 full-size ndarray를 공통 형식으로 변환"""
        if isinstance(mask, dict):
            mask_array = (np.asarray(mask['mask']) > 0).astype(np.uint8)
            return {
                'mask': mask_array,
                'offset_x': int(mask.get('offset_x', 0)),
                'offset_y': int(mask.get('offset_y', 0)),
                'height': int(mask_array.shape[0]),
                'width': int(mask_array.shape[1]),
            }
        
        mask_array = (np.asarray(mask) > 0).astype(np.uint8)
        ys, xs = np.where(mask_array > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        
        x_min, x_max = int(xs.min()), int(xs.max()) + 1
        y_min, y_max = int(ys.min()), int(ys.max()) + 1
        cropped = mask_array[y_min:y_max, x_min:x_max]
        return {
            'mask': cropped,
            'offset_x': x_min,
            'offset_y': y_min,
            'height': int(cropped.shape[0]),
            'width': int(cropped.shape[1]),
        }
    
    def _blend_overlay(self, preview: np.ndarray, mask_layer: np.ndarray) -> np.ndarray:
        """RGB preview 위에 마스크를 반투명하게 합성"""
        overlay = preview.copy()
        mask_pixels = mask_layer > 0
        if not np.any(mask_pixels):
            return overlay
        
        alpha = float(np.clip(self.config.alpha, 0.0, 1.0))
        mask_color = self._as_rgb_color(self.config.mask_color)
        overlay[mask_pixels] = (
            overlay[mask_pixels].astype(np.float32) * (1.0 - alpha)
            + mask_color.astype(np.float32) * alpha
        ).astype(np.uint8)
        
        if self.config.draw_contours:
            contour_color = tuple(int(v) for v in self._as_rgb_color(self.config.contour_color))
            contours, _ = cv2.findContours(mask_layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, contour_color, thickness=1)
        
        return overlay
    
    @staticmethod
    def _as_rgb_color(color: List[int]) -> np.ndarray:
        """YAML 색상값을 RGB uint8 배열로 변환"""
        if len(color) < 3:
            color = [255, 0, 0]
        return np.asarray(color[:3], dtype=np.uint8)
