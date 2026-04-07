"""Multi-GPU 모델 관리"""
import os
import shutil
import subprocess
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
        self._apply_cuda_visible_devices()
        self.torch = self._import_torch()
        self._log_cuda_diagnostics()
        
        self.device_ids = self._resolve_device_ids(gpu_config.device_ids)
        self.inference_devices = self._build_inference_devices()
        self.use_multi_gpu = gpu_config.use_multi_gpu and len(self.inference_devices) > 1
        
        # 주 디바이스
        self.primary_device = self.inference_devices[0]
        
        logger.info(f"Available CUDA GPUs: {self._cuda_device_count()}")
        logger.info(f"Using inference devices: {self.inference_devices}")
        logger.info(f"Primary device: {self.primary_device}")
        
        self.models: Dict[str, Any] = {}
        self._load_models()

    def _apply_cuda_visible_devices(self) -> None:
        """YAML에서 지정한 CUDA_VISIBLE_DEVICES를 torch import 전에 적용"""
        visible_devices = self.gpu_config.cuda_visible_devices
        if visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
            logger.info(f"CUDA_VISIBLE_DEVICES set from config: {visible_devices}")

    @staticmethod
    def _import_torch():
        """torch를 지연 import해서 CUDA visibility 설정이 먼저 적용되게 한다."""
        try:
            import torch
            return torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is not installed. Install the CUDA-enabled torch build "
                "before running GPU inference."
            ) from exc

    def _cuda_is_available(self) -> bool:
        return bool(self.torch.cuda.is_available())

    def _cuda_device_count(self) -> int:
        if not self._cuda_is_available():
            return 0
        return int(self.torch.cuda.device_count())

    def _log_cuda_diagnostics(self) -> None:
        """CUDA 인식 상태를 명확히 로깅"""
        logger.info("=== CUDA Diagnostics ===")
        logger.info(f"Requested device mode: {self.gpu_config.device}")
        logger.info(f"require_cuda: {self.gpu_config.require_cuda}")
        logger.info(f"allow_cpu_fallback: {self.gpu_config.allow_cpu_fallback}")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
        logger.info(f"PyTorch: {getattr(self.torch, '__version__', '<unknown>')}")
        logger.info(f"PyTorch CUDA build: {self.torch.version.cuda}")
        logger.info(f"torch.cuda.is_available(): {self.torch.cuda.is_available()}")
        logger.info(f"torch.cuda.device_count(): {self.torch.cuda.device_count()}")
        
        nvidia_smi_path = shutil.which("nvidia-smi")
        if nvidia_smi_path:
            try:
                completed = subprocess.run(
                    [
                        nvidia_smi_path,
                        "--query-gpu=index,name,memory.total,driver_version",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                if completed.returncode == 0:
                    logger.info("nvidia-smi GPUs:")
                    for line in completed.stdout.strip().splitlines():
                        logger.info(f"  {line}")
                else:
                    logger.warning(f"nvidia-smi failed: {completed.stderr.strip()}")
            except Exception as exc:
                logger.warning(f"nvidia-smi diagnostics failed: {exc}")
        else:
            logger.warning("nvidia-smi not found in PATH")
        
        logger.info("=== End CUDA Diagnostics ===")

    def _resolve_device_ids(self, device_ids: List[int]) -> List[int]:
        """사용 가능한 CUDA 디바이스 ID만 반환"""
        requested_device = self.gpu_config.device.lower()
        if requested_device == "cpu":
            return []
        
        if not self._cuda_is_available():
            message = self._build_cuda_unavailable_message()
            if (
                requested_device != "auto"
                and self.gpu_config.require_cuda
                and not self.gpu_config.allow_cpu_fallback
            ):
                raise RuntimeError(message)
            logger.warning(message)
            return []
        
        available_count = self._cuda_device_count()
        resolved = [device_id for device_id in device_ids if 0 <= device_id < available_count]
        
        if not resolved and available_count > 0:
            if self.gpu_config.require_cuda:
                raise RuntimeError(
                    f"Configured GPU IDs {device_ids} are unavailable. "
                    f"PyTorch sees {available_count} CUDA device(s). "
                    "Check gpu.device_ids and CUDA_VISIBLE_DEVICES."
                )
            logger.warning("Configured GPU IDs are unavailable; falling back to cuda:0")
            return [0]
        
        skipped = sorted(set(device_ids) - set(resolved))
        if skipped:
            logger.warning(
                f"Skipping unavailable GPU IDs: {skipped}. "
                f"PyTorch sees {available_count} CUDA device(s)."
            )
        
        return resolved

    def _build_cuda_unavailable_message(self) -> str:
        """CUDA 미인식 시 사용자가 바로 확인할 수 있는 오류 메시지"""
        return (
            "CUDA is not available to PyTorch, but GPU inference is requested. "
            f"torch.__version__={getattr(self.torch, '__version__', '<unknown>')}, "
            f"torch.version.cuda={self.torch.version.cuda}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}. "
            "Install a CUDA-enabled PyTorch build and make sure the NVIDIA driver/container "
            "runtime exposes GPUs. To intentionally run on CPU, set gpu.device='cpu', "
            "gpu.require_cuda=false, and gpu.allow_cpu_fallback=true in YAML."
        )

    def _build_inference_devices(self) -> List[str]:
        """실제 추론에 사용할 디바이스 문자열 목록"""
        requested_device = self.gpu_config.device.lower()
        if requested_device == "cpu":
            logger.warning("GPU disabled by config: gpu.device=cpu")
            return ["cpu"]
        
        if self._cuda_is_available() and self.device_ids:
            return [f"cuda:{device_id}" for device_id in self.device_ids]
        
        if self.gpu_config.allow_cpu_fallback or requested_device == "auto":
            logger.warning("Falling back to CPU inference")
            return ["cpu"]
        
        raise RuntimeError(self._build_cuda_unavailable_message())
    
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
        if self._cuda_is_available():
            for device_id in self.device_ids:
                props = self.torch.cuda.get_device_properties(device_id)
                memory_total = props.total_memory / 1024**3  # GB
                memory_allocated = self.torch.cuda.memory_allocated(device_id) / 1024**3
                logger.info(f"GPU {device_id}: {props.name}, "
                           f"Memory: {memory_allocated:.2f}GB / {memory_total:.2f}GB")
    
    def inference(
        self,
        image: Any,
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
        
        with self.torch.inference_mode():
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
        if self._cuda_is_available():
            return f"cuda:{device_id}"
        if self.gpu_config.allow_cpu_fallback or self.gpu_config.device.lower() in {"cpu", "auto"}:
            return "cpu"
        raise RuntimeError(self._build_cuda_unavailable_message())
    
    def clear_gpu_cache(self) -> None:
        """GPU 캐시 정리"""
        if self._cuda_is_available():
            for device_id in self.device_ids:
                with self.torch.cuda.device(device_id):
                    self.torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    
    def __del__(self):
        """정리"""
        try:
            self.clear_gpu_cache()
        except Exception:
            pass
