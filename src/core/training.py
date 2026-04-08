"""YOLOv8-Seg fine-tuning entrypoints."""
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ..utils.config import PipelineConfig, TrainingConfig
from ..utils.logger import logger


TRAIN_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class YOLOSegFineTuner:
    """Fine-tune an Ultralytics YOLOv8 segmentation model."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.training_config: TrainingConfig = config.training
        self._apply_cuda_visible_devices()

    def _apply_cuda_visible_devices(self) -> None:
        visible_devices = self.config.gpu.cuda_visible_devices
        if visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
            logger.info(f"CUDA_VISIBLE_DEVICES set from config: {visible_devices}")

    def prepare_dataset_yaml(self) -> Path:
        """Return a dataset YAML path, generating one when only dirs are configured."""
        if self.training_config.dataset_yaml:
            dataset_yaml = Path(self.training_config.dataset_yaml)
            if not dataset_yaml.exists():
                raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
            return dataset_yaml

        train_images = Path(self.training_config.train_images)
        val_images = Path(self.training_config.val_images)
        train_labels = Path(self.training_config.train_labels)
        val_labels = Path(self.training_config.val_labels)

        self._validate_yolo_seg_dataset(train_images, val_images, train_labels, val_labels)

        output_dir = Path(self.config.data.output_dir) / "training"
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_yaml = output_dir / "dataset.yaml"

        data = {
            "train": self._path_for_yaml(train_images),
            "val": self._path_for_yaml(val_images),
            "names": {index: name for index, name in enumerate(self.training_config.class_names)},
        }
        with open(dataset_yaml, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)

        logger.info(f"Generated training dataset YAML: {dataset_yaml}")
        return dataset_yaml

    def train(self) -> Dict[str, Any]:
        """Run YOLOv8-Seg fine-tuning and return key output paths."""
        from ultralytics import YOLO

        dataset_yaml = self.prepare_dataset_yaml()
        weights = self.training_config.pretrained_weights or self.config.model.model_weights
        train_args = self._build_train_args(dataset_yaml)

        logger.info(f"Starting YOLOv8-Seg fine-tuning from weights: {weights}")
        logger.info(f"Training dataset YAML: {dataset_yaml}")
        logger.info(f"Training device: {train_args.get('device')}")

        model = YOLO(weights, task="segment")
        results = model.train(**train_args)

        result_save_dir = getattr(results, "save_dir", None)
        run_dir = Path(result_save_dir) if result_save_dir else (
            Path(self.training_config.project) / self.training_config.name
        )
        best_model = run_dir / "weights" / "best.pt"
        last_model = run_dir / "weights" / "last.pt"

        logger.info(f"Fine-tuning complete: run_dir={run_dir}")
        if best_model.exists():
            logger.info(f"Best weights: {best_model}")
        if last_model.exists():
            logger.info(f"Last weights: {last_model}")

        return {
            "results": results,
            "dataset_yaml": dataset_yaml,
            "run_dir": run_dir,
            "best_model": best_model,
            "last_model": last_model,
        }

    def _build_train_args(self, dataset_yaml: Path) -> Dict[str, Any]:
        cfg = self.training_config
        return {
            "data": str(dataset_yaml),
            "epochs": cfg.epochs,
            "imgsz": cfg.imgsz,
            "batch": cfg.batch,
            "workers": cfg.workers,
            "device": self._resolve_train_device(),
            "project": cfg.project,
            "name": cfg.name,
            "exist_ok": cfg.exist_ok,
            "resume": cfg.resume,
            "patience": cfg.patience,
            "optimizer": cfg.optimizer,
            "lr0": cfg.lr0,
            "lrf": cfg.lrf,
            "momentum": cfg.momentum,
            "weight_decay": cfg.weight_decay,
            "warmup_epochs": cfg.warmup_epochs,
            "seed": cfg.seed,
            "deterministic": cfg.deterministic,
            "amp": cfg.amp,
            "cache": cfg.cache,
            "rect": cfg.rect,
            "val": cfg.val,
            "plots": cfg.plots,
            "save_period": cfg.save_period,
            "single_cls": cfg.single_cls,
            "close_mosaic": cfg.close_mosaic,
            "dropout": cfg.dropout,
            "overlap_mask": cfg.overlap_mask,
            "mask_ratio": cfg.mask_ratio,
            "hsv_h": cfg.hsv_h,
            "hsv_s": cfg.hsv_s,
            "hsv_v": cfg.hsv_v,
            "degrees": cfg.degrees,
            "translate": cfg.translate,
            "scale": cfg.scale,
            "shear": cfg.shear,
            "perspective": cfg.perspective,
            "flipud": cfg.flipud,
            "fliplr": cfg.fliplr,
            "mosaic": cfg.mosaic,
            "mixup": cfg.mixup,
            "copy_paste": cfg.copy_paste,
            "erasing": cfg.erasing,
        }

    def _resolve_train_device(self) -> str:
        cfg = self.training_config
        if cfg.device:
            return str(cfg.device)

        gpu_cfg = self.config.gpu
        requested_device = gpu_cfg.device.lower()
        if requested_device == "cpu":
            return "cpu"

        try:
            import torch
        except ImportError as exc:
            if gpu_cfg.require_cuda and not gpu_cfg.allow_cpu_fallback:
                raise RuntimeError(
                    "PyTorch is not installed, but CUDA training is requested."
                ) from exc
            logger.warning("PyTorch is not installed; falling back to CPU training")
            return "cpu"

        if torch.cuda.is_available():
            available_count = torch.cuda.device_count()
            device_ids = [
                device_id
                for device_id in gpu_cfg.device_ids
                if 0 <= device_id < available_count
            ]
            if device_ids:
                return ",".join(str(device_id) for device_id in device_ids)

        if requested_device == "auto" or gpu_cfg.allow_cpu_fallback:
            logger.warning("CUDA is unavailable for training; falling back to CPU")
            return "cpu"

        raise RuntimeError(
            "CUDA is not available to PyTorch, but GPU training is requested. "
            "Set training.device='cpu' or gpu.allow_cpu_fallback=true to train on CPU."
        )

    def _validate_yolo_seg_dataset(
        self,
        train_images: Path,
        val_images: Path,
        train_labels: Path,
        val_labels: Path,
    ) -> None:
        for path, label in [
            (train_images, "training images"),
            (val_images, "validation images"),
            (train_labels, "training labels"),
            (val_labels, "validation labels"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} directory not found: {path}")
            if not path.is_dir():
                raise NotADirectoryError(f"{label} path is not a directory: {path}")

        train_count = self._count_images(train_images)
        val_count = self._count_images(val_images)
        if train_count == 0:
            raise ValueError(f"No training images found in {train_images}")
        if val_count == 0:
            raise ValueError(f"No validation images found in {val_images}")

        missing_train = self._missing_label_files(train_images, train_labels)
        missing_val = self._missing_label_files(val_images, val_labels)
        missing_count = len(missing_train) + len(missing_val)
        if missing_count:
            examples = (missing_train + missing_val)[:5]
            raise ValueError(
                f"{missing_count} image(s) do not have matching YOLO label files. "
                f"Examples: {examples}"
            )

        logger.info(
            f"YOLO segmentation dataset validated: "
            f"train_images={train_count}, val_images={val_count}"
        )

    @staticmethod
    def _count_images(image_dir: Path) -> int:
        return sum(
            1
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in TRAIN_IMAGE_EXTENSIONS
        )

    @staticmethod
    def _missing_label_files(image_dir: Path, label_dir: Path) -> List[str]:
        missing = []
        for image_path in image_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in TRAIN_IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing.append(str(label_path))
        return missing

    @staticmethod
    def _path_for_yaml(path: Path) -> str:
        return path.resolve().as_posix()
