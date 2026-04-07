"""Multi-GPU 모델 관리"""
import torch
from typing import Optional, Dict, Any, List

from ..utils.logger import logger
from ..utils.config import GPUConfig, ModelConfig


class MultiGPUModelManager:
    """Multi-GPU 모델 관리 및 로드"""
    
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
        self.device_ids = self._resolve_device_ids(gpu_config.device_ids)
        self.inference_devices = self._build_inference_devices()
        self.use_multi_gpu = gpu_config.use_multi_gpu and len(self.inference_devices) > 1
        
        # 주 디바이스
        self.primary_device = self.inference_devices[0]
        
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        logger.info(f"Using inference devices: {self.inference_devices}")
        logger.info(f"Primary device: {self.primary_device}")
        
        self.models: Dict[str, Any] = {}
        self._load_models()

    @staticmethod
    def _resolve_device_ids(device_ids: List[int]) -> List[int]:
        """사용 가능한 CUDA 디바이스 ID만 반환"""
        if not torch.cuda.is_available():
            return []
        
        available_count = torch.cuda.device_count()
        resolved = [device_id for device_id in device_ids if 0 <= device_id < available_count]
        
        if not resolved and available_count > 0:
            logger.warning("Configured GPU IDs are unavailable; falling back to cuda:0")
            return [0]
        
        skipped = sorted(set(device_ids) - set(resolved))
        if skipped:
            logger.warning(f"Skipping unavailable GPU IDs: {skipped}")
        
        return resolved

    def _build_inference_devices(self) -> List[str]:
        """실제 추론에 사용할 디바이스 문자열 목록"""
        if torch.cuda.is_available() and self.device_ids:
            return [f"cuda:{device_id}" for device_id in self.device_ids]
        return ["cpu"]
    
    def _load_models(self) -> None:
        """YOLOv8 모델을 추론 디바이스별로 로드"""
        try:
            from ultralytics import YOLO
            
            for device in self.inference_devices:
                logger.info(f"Loading model: {self.model_config.model_name} on {device}")
                
                model = YOLO(self.model_config.model_weights)
                model.to(device)
                self.models[device] = model
                
                logger.info(f"Model loaded successfully on {device}")
            
            # GPU 메모리 정보
            self._log_gpu_memory()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _log_gpu_memory(self) -> None:
        """GPU 메모리 정보 출력"""
        if torch.cuda.is_available():
            for device_id in self.device_ids:
                props = torch.cuda.get_device_properties(device_id)
                memory_total = props.total_memory / 1024**3  # GB
                memory_allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                logger.info(f"GPU {device_id}: {props.name}, "
                           f"Memory: {memory_allocated:.2f}GB / {memory_total:.2f}GB")
    
    def inference(
        self,
        image: torch.Tensor,
        conf: float = 0.4,
        iou: float = 0.5,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """단일 이미지 추론
        
        Args:
            image: 입력 이미지 (HxWxC) numpy 또는 tensor
            conf: 신뢰도 임계값
            iou: IOU 임계값
            device: 추론 디바이스. None이면 primary device 사용
            
        Returns:
            추론 결과 (마스크, 바운딩박스 등)
        """
        target_device = device or self.primary_device
        if target_device not in self.models:
            raise ValueError(f"Model is not loaded on device: {target_device}")
        
        model = self.models[target_device]
        
        with torch.inference_mode():
            results = model(
                image,
                conf=conf,
                iou=iou,
                device=target_device,
                verbose=False,
            )
        
        return self._parse_yolov8_results(results[0])
    
    def _parse_yolov8_results(self, result) -> Dict[str, Any]:
        """YOLOv8 결과 파싱
        
        Returns:
            {
                'masks': np.ndarray (N, H, W),
                'boxes': np.ndarray (N, 4),
                'scores': np.ndarray (N,),
                'class_ids': np.ndarray (N,),
            }
        """
        result_dict = {
            'masks': None,
            'boxes': None,
            'scores': None,
            'class_ids': None,
        }
        
        if result.masks is not None:
            result_dict['masks'] = result.masks.data.cpu().numpy()  # (N, H, W)
        
        if result.boxes is not None:
            result_dict['boxes'] = result.boxes.xyxy.cpu().numpy()  # (N, 4)
            result_dict['scores'] = result.boxes.conf.cpu().numpy()  # (N,)
            result_dict['class_ids'] = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
        
        return result_dict

    def get_inference_devices(self) -> List[str]:
        """추론에 사용할 디바이스 문자열 목록"""
        if self.use_multi_gpu:
            return self.inference_devices
        return [self.primary_device]
    
    def get_device(self, device_id: Optional[int] = None) -> str:
        """디바이스 문자열 반환
        
        Args:
            device_id: 디바이스 ID (None일 경우 primary device)
            
        Returns:
            디바이스 문자열 (예: "cuda:0")
        """
        if device_id is None:
            return self.primary_device
        return f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    
    def clear_gpu_cache(self) -> None:
        """GPU 캐시 정리"""
        if torch.cuda.is_available():
            for device_id in self.device_ids:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    
    def __del__(self):
        """정리"""
        self.clear_gpu_cache()
