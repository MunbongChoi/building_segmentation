"""SAHI 기반 슬라이싱 추론 엔진"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

from ..utils.logger import logger
from ..utils.config import ModelConfig, GPUConfig
from .model_manager import MultiGPUModelManager


class SAHIInferenceEngine:
    """SAHI 기반 추론 엔진 (멀티-GPU 지원)"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        gpu_config: GPUConfig,
    ):
        """
        Args:
            model_config: 모델 설정
            gpu_config: GPU 설정
        """
        self.model_config = model_config
        self.gpu_config = gpu_config
        self.model_manager = MultiGPUModelManager(model_config, gpu_config)
        
        self.slice_height = model_config.sahi_slice_height
        self.slice_width = model_config.sahi_slice_width
        self.overlap_height_ratio = model_config.sahi_overlap_height_ratio
        self.overlap_width_ratio = model_config.sahi_overlap_width_ratio
        if self.slice_height <= 0 or self.slice_width <= 0:
            raise ValueError("SAHI slice size must be positive")
        if not 0 <= self.overlap_height_ratio < 1 or not 0 <= self.overlap_width_ratio < 1:
            raise ValueError("SAHI overlap ratios must be in the range [0, 1)")
        
        logger.info(f"SAHI Engine initialized: {self.slice_height}x{self.slice_width}, "
                   f"overlap: {self.overlap_height_ratio:.1%}x{self.overlap_width_ratio:.1%}")
    
    def predict_with_slicing(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        devices: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """슬라이싱을 이용한 추론
        
        Args:
            image: 입력 이미지 (H, W, 3) uint8 또는 float
            conf_threshold: 신뢰도 임계값
            iou_threshold: IOU 임계값
            devices: 이 호출에서 사용할 디바이스 목록. None이면 전체 inference device 사용
            
        Returns:
            {
                'masks': List[Dict],  # crop mask + 전역 픽셀 offset
                'boxes': np.ndarray (N, 4),  # 바운딩박스
                'scores': np.ndarray (N,),
                'class_ids': np.ndarray (N,),
                'image_height': int,
                'image_width': int,
            }
        """
        image_height, image_width = image.shape[:2]
        
        logger.info(f"Starting SAHI inference on {image_width}x{image_height} image")
        
        # 슬라이스 생성
        slices = self._generate_slices(image_height, image_width)
        logger.info(f"Generated {len(slices)} slices")
        
        # 각 슬라이스에 대해 추론 수행
        all_results = self._predict_slices(
            image,
            slices,
            conf_threshold,
            iou_threshold,
            devices=devices,
        )
        
        # 결과 병합
        merged_result = self._merge_predictions(
            all_results,
            image_height,
            image_width,
            self.model_config.mask_nms_threshold or iou_threshold,
        )
        
        logger.info(f"Inference complete: {len(merged_result['boxes'])} objects detected")
        
        return merged_result

    def _predict_slices(
        self,
        image: np.ndarray,
        slices: List[Dict[str, int]],
        conf_threshold: float,
        iou_threshold: float,
        devices: Optional[List[str]] = None,
    ) -> List[Tuple[Dict, Dict]]:
        """슬라이스 목록을 단일 또는 멀티 GPU로 추론"""
        devices = devices or self.model_manager.get_inference_devices()
        max_workers = min(len(devices), self.gpu_config.num_workers, len(slices))
        
        if max_workers <= 1:
            all_results = []
            for slice_info in slices:
                result = self._predict_slice(
                    image,
                    slice_info,
                    conf_threshold,
                    iou_threshold,
                    device=devices[0],
                )
                all_results.append((slice_info, result))
            return all_results
        
        logger.info(f"Running sliced inference with {max_workers} workers on {devices}")
        all_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_slice = {}
            for index, slice_info in enumerate(slices):
                device = devices[index % len(devices)]
                future = executor.submit(
                    self._predict_slice,
                    image,
                    slice_info,
                    conf_threshold,
                    iou_threshold,
                    device,
                )
                future_to_slice[future] = slice_info
            
            for future in as_completed(future_to_slice):
                slice_info = future_to_slice[future]
                result = future.result()
                all_results.append((slice_info, result))
        
        return all_results
    
    def _generate_slices(
        self,
        image_height: int,
        image_width: int,
    ) -> List[Dict[str, int]]:
        """슬라이스 위치 생성
        
        Returns:
            [{'x': int, 'y': int, 'width': int, 'height': int}, ...]
        """
        slices = []
        
        # 스트라이드 계산
        stride_height = max(1, int(self.slice_height * (1 - self.overlap_height_ratio)))
        stride_width = max(1, int(self.slice_width * (1 - self.overlap_width_ratio)))
        
        y_offsets = self._axis_offsets(image_height, self.slice_height, stride_height)
        x_offsets = self._axis_offsets(image_width, self.slice_width, stride_width)
        
        for y in y_offsets:
            for x in x_offsets:
                slice_height = min(self.slice_height, image_height - y)
                slice_width = min(self.slice_width, image_width - x)
                
                slices.append({
                    'x': x,
                    'y': y,
                    'width': slice_width,
                    'height': slice_height,
                })
        
        return slices

    @staticmethod
    def _axis_offsets(length: int, window_size: int, stride: int) -> List[int]:
        """축 하나에 대해 끝단을 보장하는 sliding-window 시작점 생성"""
        if length <= window_size:
            return [0]
        
        offsets = list(range(0, length - window_size + 1, stride))
        last_offset = length - window_size
        if offsets[-1] != last_offset:
            offsets.append(last_offset)
        
        return offsets
    
    def _predict_slice(
        self,
        image: np.ndarray,
        slice_info: Dict[str, int],
        conf_threshold: float,
        iou_threshold: float,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """단일 슬라이스 추론"""
        x, y = slice_info['x'], slice_info['y']
        width, height = slice_info['width'], slice_info['height']
        
        # 슬라이스 추출
        slice_image = image[y:y+height, x:x+width].copy()
        
        # 추론
        result = self.model_manager.inference(
            slice_image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
        )
        
        # 슬라이스 오프셋 추가
        result['slice_offset_x'] = x
        result['slice_offset_y'] = y
        
        return result

    def offset_result(
        self,
        result: Dict[str, Any],
        offset_x: int,
        offset_y: int,
        image_height: int,
        image_width: int,
    ) -> Dict[str, Any]:
        """타일 좌표계의 추론 결과를 전체 이미지 좌표계로 이동"""
        masks = [
            self._offset_mask_record(mask, offset_x, offset_y)
            for mask in (result.get('masks') or [])
        ]
        
        boxes = result.get('boxes')
        if boxes is not None and len(boxes) > 0:
            boxes = np.asarray(boxes) + np.array([offset_x, offset_y, offset_x, offset_y])
        else:
            boxes = np.empty((0, 4))
        
        scores = result.get('scores')
        class_ids = result.get('class_ids')
        
        return {
            'masks': masks,
            'boxes': boxes,
            'scores': np.asarray(scores if scores is not None else []),
            'class_ids': np.asarray(class_ids if class_ids is not None else [], dtype=int),
            'image_height': image_height,
            'image_width': image_width,
        }

    def nms_predictions(
        self,
        masks: List[Any],
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        image_height: int,
        image_width: int,
        iou_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """여러 타일/슬라이스 결과에 전역 마스크 NMS 적용"""
        if iou_threshold is None:
            iou_threshold = self.model_config.mask_nms_threshold
        
        if len(masks) > 0:
            masks, boxes, scores, class_ids = self._apply_mask_nms(
                masks,
                np.asarray(boxes),
                np.asarray(scores),
                np.asarray(class_ids, dtype=int),
                iou_threshold,
            )
        
        return {
            'masks': masks,
            'boxes': np.asarray(boxes) if len(boxes) else np.empty((0, 4)),
            'scores': np.asarray(scores) if len(scores) else np.empty(0),
            'class_ids': np.asarray(class_ids, dtype=int) if len(class_ids) else np.empty(0, dtype=int),
            'image_height': image_height,
            'image_width': image_width,
        }

    def get_inference_devices(self) -> List[str]:
        """추론에 사용할 디바이스 문자열 목록"""
        return self.model_manager.get_inference_devices()
    
    def _merge_predictions(
        self,
        all_results: List[Tuple[Dict, Dict]],
        image_height: int,
        image_width: int,
        iou_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """모든 슬라이스 결과를 병합 및 NMS 적용"""
        merged_masks = []
        merged_boxes = []
        merged_scores = []
        merged_class_ids = []
        
        # 결과 수집
        for slice_info, result in all_results:
            masks = result.get('masks')
            boxes = result.get('boxes')
            scores = result.get('scores')
            class_ids = result.get('class_ids')
            
            if masks is None or len(masks) == 0:
                continue
            if boxes is None or scores is None or class_ids is None:
                continue
            
            offset_x = result['slice_offset_x']
            offset_y = result['slice_offset_y']
            
            for i, mask in enumerate(masks):
                if i >= len(boxes) or i >= len(scores) or i >= len(class_ids):
                    continue
                
                slice_h, slice_w = slice_info['height'], slice_info['width']
                mask_record = self._make_mask_record(
                    mask,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    slice_height=slice_h,
                    slice_width=slice_w,
                )
                if mask_record is None:
                    continue
                
                merged_masks.append(mask_record)
                merged_boxes.append(boxes[i] + [offset_x, offset_y, offset_x, offset_y])
                merged_scores.append(scores[i])
                merged_class_ids.append(class_ids[i])
        
        # NMS 적용
        if len(merged_boxes) > 0:
            merged_masks, merged_boxes, merged_scores, merged_class_ids = self._apply_mask_nms(
                merged_masks,
                merged_boxes,
                merged_scores,
                merged_class_ids,
                iou_threshold,
            )
        
        return {
            'masks': merged_masks,
            'boxes': np.array(merged_boxes) if merged_boxes else np.empty((0, 4)),
            'scores': np.array(merged_scores) if merged_scores else np.empty(0),
            'class_ids': np.array(merged_class_ids) if merged_class_ids else np.empty(0, dtype=int),
            'image_height': image_height,
            'image_width': image_width,
        }
    
    def _apply_mask_nms(
        self,
        masks: List[Any],
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        iou_threshold: float,
    ) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray]:
        """마스크 기반 NMS 적용"""
        if len(masks) == 0:
            return [], np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
        
        boxes = np.asarray(boxes)
        scores = np.asarray(scores)
        class_ids = np.asarray(class_ids, dtype=int)
        
        # 신뢰도 기준 정렬
        sorted_indices = np.argsort(scores)[::-1]
        
        keep_indices = []
        for i in sorted_indices:
            # 기존 keep 마스크들과 IOU 계산
            is_duplicate = False
            for keep_idx in keep_indices:
                iou = self._mask_iou(masks[i], masks[keep_idx])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep_indices.append(i)
        
        # 필터링된 결과 반환
        filtered_masks = [masks[i] for i in keep_indices]
        filtered_boxes = boxes[keep_indices]
        filtered_scores = scores[keep_indices]
        filtered_class_ids = class_ids[keep_indices]
        
        return filtered_masks, filtered_boxes, filtered_scores, filtered_class_ids
    
    @staticmethod
    def _make_mask_record(
        mask: np.ndarray,
        offset_x: int,
        offset_y: int,
        slice_height: int,
        slice_width: int,
    ) -> Optional[Dict[str, Any]]:
        """전체 이미지 마스크 대신 crop mask와 전역 offset만 저장"""
        if mask.shape[:2] != (slice_height, slice_width):
            mask = cv2.resize(mask, (slice_width, slice_height), interpolation=cv2.INTER_NEAREST)
        else:
            mask = mask[:slice_height, :slice_width]
        
        binary_mask = (mask > 0.5).astype(np.uint8)
        ys, xs = np.where(binary_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        
        x_min, x_max = int(xs.min()), int(xs.max()) + 1
        y_min, y_max = int(ys.min()), int(ys.max()) + 1
        cropped_mask = binary_mask[y_min:y_max, x_min:x_max]
        
        return {
            'mask': cropped_mask,
            'offset_x': int(offset_x + x_min),
            'offset_y': int(offset_y + y_min),
            'height': int(cropped_mask.shape[0]),
            'width': int(cropped_mask.shape[1]),
        }

    @staticmethod
    def _offset_mask_record(mask: Any, offset_x: int, offset_y: int) -> Any:
        """crop mask record의 offset을 추가 이동"""
        if not isinstance(mask, dict):
            return mask
        
        shifted = dict(mask)
        shifted['offset_x'] = int(shifted.get('offset_x', 0) + offset_x)
        shifted['offset_y'] = int(shifted.get('offset_y', 0) + offset_y)
        return shifted

    @staticmethod
    def _normalize_mask_record(mask: Any) -> Optional[Dict[str, Any]]:
        """기존 full-size mask와 crop mask record를 같은 표현으로 정규화"""
        if isinstance(mask, dict):
            mask_array = np.asarray(mask['mask'])
            return {
                'mask': (mask_array > 0).astype(np.uint8),
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
        cropped_mask = mask_array[y_min:y_max, x_min:x_max]
        
        return {
            'mask': cropped_mask,
            'offset_x': x_min,
            'offset_y': y_min,
            'height': int(cropped_mask.shape[0]),
            'width': int(cropped_mask.shape[1]),
        }

    @classmethod
    def _mask_iou(cls, mask1: Any, mask2: Any) -> float:
        """두 마스크 간의 IOU 계산"""
        record1 = cls._normalize_mask_record(mask1)
        record2 = cls._normalize_mask_record(mask2)
        if record1 is None or record2 is None:
            return 0.0
        
        x1_min = record1['offset_x']
        y1_min = record1['offset_y']
        x1_max = x1_min + record1['width']
        y1_max = y1_min + record1['height']
        
        x2_min = record2['offset_x']
        y2_min = record2['offset_y']
        x2_max = x2_min + record2['width']
        y2_max = y2_min + record2['height']
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        area1 = int((record1['mask'] > 0).sum())
        area2 = int((record2['mask'] > 0).sum())
        
        if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
            return 0.0
        
        crop1 = record1['mask'][
            inter_y_min - y1_min:inter_y_max - y1_min,
            inter_x_min - x1_min:inter_x_max - x1_min,
        ]
        crop2 = record2['mask'][
            inter_y_min - y2_min:inter_y_max - y2_min,
            inter_x_min - x2_min:inter_x_max - x2_min,
        ]
        
        intersection = int(np.logical_and(crop1, crop2).sum())
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def clear_gpu_cache(self) -> None:
        """GPU 캐시 정리"""
        self.model_manager.clear_gpu_cache()
