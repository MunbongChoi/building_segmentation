"""YOLOv8-Seg fine-tuning entrypoints."""
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..utils.config import PipelineConfig, TrainingConfig
from ..utils.logger import logger


TRAIN_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MASK_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_FORMAT_MASK_PNG = "mask_png"
LABEL_FORMAT_YOLO_TXT = "yolo_txt"


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
            return self._resolve_existing_dataset_yaml(dataset_yaml)

        label_format = self.training_config.label_format.lower()
        if label_format == LABEL_FORMAT_MASK_PNG:
            return self._prepare_mask_png_dataset_yaml()
        if label_format != LABEL_FORMAT_YOLO_TXT:
            raise ValueError(
                "training.label_format must be either 'mask_png' or 'yolo_txt'. "
                f"Got: {self.training_config.label_format}"
            )

        train_images = self._required_path(
            self.training_config.train_images,
            "training.train_images",
        )
        train_labels = self._required_path(
            self.training_config.train_labels,
            "training.train_labels",
        )
        val_images = self._optional_path(self.training_config.val_images)
        val_labels = self._optional_path(self.training_config.val_labels)

        val_images, val_labels = self._validate_yolo_seg_dataset(
            train_images,
            val_images,
            train_labels,
            val_labels,
        )

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

    def _prepare_mask_png_dataset_yaml(self) -> Path:
        """Convert masked PNG labels into a YOLOv8-Seg dataset."""
        train_images = self._required_path(
            self.training_config.train_images,
            "training.train_images",
        )
        train_masks = self._resolve_mask_dir(
            self.training_config.train_masks,
            self.training_config.train_labels,
            "training.train_masks",
        )
        val_images = self._optional_path(self.training_config.val_images)
        val_masks = self._optional_path(
            self.training_config.val_masks
            if self.training_config.val_masks is not None
            else self.training_config.val_labels
        )

        self._validate_image_mask_dataset(train_images, train_masks, split_name="train")
        val_images, val_masks = self._resolve_mask_validation_dirs(
            train_images=train_images,
            train_masks=train_masks,
            val_images=val_images,
            val_masks=val_masks,
        )
        self._validate_image_mask_dataset(val_images, val_masks, split_name="val")

        dataset_root = Path(self.training_config.mask_dataset_dir)
        self._materialize_mask_png_split(
            image_dir=train_images,
            mask_dir=train_masks,
            dataset_root=dataset_root,
            split_name="train",
        )
        self._materialize_mask_png_split(
            image_dir=val_images,
            mask_dir=val_masks,
            dataset_root=dataset_root,
            split_name="val",
        )

        dataset_yaml = dataset_root / "dataset.yaml"
        data = {
            "path": self._path_for_yaml(dataset_root),
            "train": "images/train",
            "val": "images/val",
            "names": {index: name for index, name in enumerate(self.training_config.class_names)},
        }
        with open(dataset_yaml, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)

        logger.info(f"Generated mask PNG training dataset YAML: {dataset_yaml}")
        return dataset_yaml

    def _resolve_mask_dir(
        self,
        mask_path_value: object,
        fallback_path_value: object,
        label: str,
    ) -> Path:
        mask_dir = self._optional_path(mask_path_value)
        if mask_dir is not None:
            return mask_dir

        fallback_dir = self._optional_path(fallback_path_value)
        if fallback_dir is not None:
            return fallback_dir

        raise ValueError(f"{label} is required when training.label_format='mask_png'.")

    def _resolve_mask_validation_dirs(
        self,
        train_images: Path,
        train_masks: Path,
        val_images: Optional[Path],
        val_masks: Optional[Path],
    ) -> Tuple[Path, Path]:
        val_ready = (
            val_images is not None
            and val_masks is not None
            and val_images.exists()
            and val_images.is_dir()
            and val_masks.exists()
            and val_masks.is_dir()
            and self._count_images(val_images) > 0
        )
        if val_ready:
            return val_images, val_masks

        if self.training_config.use_train_as_val_if_missing:
            logger.warning(
                "Validation mask dataset is missing or empty; using training images/masks "
                "as validation. Metrics will be optimistic."
            )
            return train_images, train_masks

        missing_paths = [
            self._format_optional_path(path, label)
            for path, label in [
                (val_images, "training.val_images"),
                (val_masks, "training.val_masks"),
            ]
            if path is None or not path.exists() or not path.is_dir()
        ]
        raise FileNotFoundError(
            "Validation mask dataset is missing or empty. "
            f"Missing/invalid paths: {missing_paths or [str(val_images)]}. "
            "Set training.use_train_as_val_if_missing=true to reuse training data."
        )

    def _validate_image_mask_dataset(
        self,
        image_dir: Path,
        mask_dir: Path,
        split_name: str,
    ) -> None:
        for path, label in [
            (image_dir, f"{split_name} images"),
            (mask_dir, f"{split_name} mask labels"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} directory not found: {path}")
            if not path.is_dir():
                raise NotADirectoryError(f"{label} path is not a directory: {path}")

        image_files = self._list_images(image_dir)
        if not image_files:
            raise ValueError(f"No {split_name} images found in {image_dir}")

        missing_masks = [
            str(image_path)
            for image_path in image_files
            if self._find_matching_mask(image_path, mask_dir) is None
        ]
        if missing_masks:
            raise ValueError(
                f"{len(missing_masks)} {split_name} image(s) do not have matching mask PNG files. "
                f"Examples: {missing_masks[:5]}"
            )

        logger.info(
            f"Mask PNG dataset validated: split={split_name}, "
            f"images={len(image_files)}, masks_dir={mask_dir}"
        )

    def _materialize_mask_png_split(
        self,
        image_dir: Path,
        mask_dir: Path,
        dataset_root: Path,
        split_name: str,
    ) -> None:
        image_output_dir = dataset_root / "images" / split_name
        label_output_dir = dataset_root / "labels" / split_name
        image_output_dir.mkdir(parents=True, exist_ok=True)
        label_output_dir.mkdir(parents=True, exist_ok=True)

        image_files = self._list_images(image_dir)
        for image_path in image_files:
            mask_path = self._find_matching_mask(image_path, mask_dir)
            if mask_path is None:
                raise FileNotFoundError(f"Mask PNG not found for image: {image_path}")

            output_image_path = image_output_dir / image_path.name
            output_label_path = label_output_dir / f"{image_path.stem}.txt"
            self._link_or_copy_image(image_path, output_image_path)
            self._write_yolo_label_from_mask_png(
                image_path=image_path,
                mask_path=mask_path,
                output_label_path=output_label_path,
            )

        logger.info(
            f"Materialized {len(image_files)} {split_name} sample(s) from mask PNG labels "
            f"under {dataset_root}"
        )

    def _write_yolo_label_from_mask_png(
        self,
        image_path: Path,
        mask_path: Path,
        output_label_path: Path,
    ) -> None:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read training image: {image_path}")
        image_height, image_width = image.shape[:2]

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask PNG: {mask_path}")

        if mask.shape[:2] != (image_height, image_width):
            if not self.training_config.resize_masks_to_images:
                raise ValueError(
                    f"Mask/image size mismatch: image={image_path} shape={image.shape[:2]}, "
                    f"mask={mask_path} shape={mask.shape[:2]}"
                )
            mask = cv2.resize(
                mask,
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            )
            logger.warning(
                f"Resized mask to image size: mask={mask_path}, size={image_width}x{image_height}"
            )

        lines = self._mask_png_to_yolo_lines(mask, image_width, image_height)
        output_label_path.parent.mkdir(parents=True, exist_ok=True)
        output_label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    def _mask_png_to_yolo_lines(self, mask, image_width: int, image_height: int) -> List[str]:
        import cv2
        import numpy as np

        lines: List[str] = []
        for instance_mask in self._iter_instance_masks(mask):
            mask_uint8 = instance_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.training_config.mask_min_area:
                    continue

                epsilon = float(self.training_config.mask_simplification_epsilon)
                polygon = cv2.approxPolyDP(contour, epsilon, closed=True)
                points = polygon.reshape(-1, 2)
                if len(points) < 3:
                    continue

                coords = []
                for x, y in points:
                    coords.append(f"{min(max(float(x) / image_width, 0.0), 1.0):.6f}")
                    coords.append(f"{min(max(float(y) / image_height, 0.0), 1.0):.6f}")

                if len(coords) >= 6:
                    lines.append("0 " + " ".join(coords))

        return lines

    def _iter_instance_masks(self, mask):
        import cv2
        import numpy as np

        threshold = int(self.training_config.mask_threshold)
        if self.training_config.mask_is_instance_encoded:
            if mask.ndim == 2:
                for value in np.unique(mask):
                    if int(value) <= threshold:
                        continue
                    yield mask == value
                return

            color_mask = mask[..., :3]
            flat_colors = color_mask.reshape(-1, color_mask.shape[-1])
            unique_colors = np.unique(flat_colors, axis=0)
            for color in unique_colors:
                if np.all(color <= threshold):
                    continue
                yield np.all(color_mask == color, axis=-1)
            return

        gray_mask = self._mask_to_grayscale(mask)
        foreground = gray_mask > threshold
        if np.any(foreground):
            yield foreground

    @staticmethod
    def _mask_to_grayscale(mask):
        import cv2

        if mask.ndim == 2:
            return mask
        if mask.shape[-1] == 4:
            return cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    def _link_or_copy_image(self, source_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            return

        if self.training_config.copy_mask_dataset_images:
            shutil.copy2(source_path, output_path)
            return

        try:
            output_path.symlink_to(source_path.resolve())
            return
        except OSError:
            pass

        try:
            os.link(source_path, output_path)
            return
        except OSError:
            shutil.copy2(source_path, output_path)

    @staticmethod
    def _list_images(image_dir: Path) -> List[Path]:
        return sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in TRAIN_IMAGE_EXTENSIONS
        )

    def _find_matching_mask(self, image_path: Path, mask_dir: Path) -> Optional[Path]:
        mask_suffix = self.training_config.mask_suffix or ""
        stem_candidates = [f"{image_path.stem}{mask_suffix}"]
        if mask_suffix:
            stem_candidates.append(image_path.stem)

        for stem in stem_candidates:
            for suffix in MASK_IMAGE_EXTENSIONS:
                candidate = mask_dir / f"{stem}{suffix}"
                if candidate.exists() and candidate.is_file():
                    return candidate
        return None

    def _resolve_existing_dataset_yaml(self, dataset_yaml: Path) -> Path:
        """Return an existing dataset YAML, adding val=train when val is absent."""
        with open(dataset_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        has_val = bool(data.get("val"))
        if has_val:
            return dataset_yaml

        if not self.training_config.use_train_as_val_if_missing:
            raise ValueError(
                f"Dataset YAML has no validation split: {dataset_yaml}. "
                "Set training.use_train_as_val_if_missing=true to reuse train as val."
            )
        if not data.get("train"):
            raise ValueError(f"Dataset YAML has neither train nor val split: {dataset_yaml}")

        data["val"] = data["train"]
        output_dir = Path(self.config.data.output_dir) / "training"
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_yaml = output_dir / "dataset_with_train_as_val.yaml"
        with open(resolved_yaml, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)

        logger.warning(
            f"Dataset YAML has no validation split; generated {resolved_yaml} with val=train."
        )
        return resolved_yaml

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
        val_images: Optional[Path],
        train_labels: Path,
        val_labels: Optional[Path],
    ) -> Tuple[Path, Path]:
        for path, label in [
            (train_images, "training images"),
            (train_labels, "training labels"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} directory not found: {path}")
            if not path.is_dir():
                raise NotADirectoryError(f"{label} path is not a directory: {path}")

        train_count = self._count_images(train_images)
        if train_count == 0:
            raise ValueError(f"No training images found in {train_images}")

        missing_train = self._missing_label_files(train_images, train_labels)
        if missing_train:
            raise ValueError(
                f"{len(missing_train)} training image(s) do not have matching YOLO label files. "
                f"Examples: {missing_train[:5]}"
            )

        val_images, val_labels = self._resolve_validation_dirs(
            train_images=train_images,
            train_labels=train_labels,
            val_images=val_images,
            val_labels=val_labels,
        )

        val_count = self._count_images(val_images)
        missing_val = self._missing_label_files(val_images, val_labels)
        if missing_val:
            raise ValueError(
                f"{len(missing_val)} validation image(s) do not have matching YOLO label files. "
                f"Examples: {missing_val[:5]}"
            )

        logger.info(
            f"YOLO segmentation dataset validated: "
            f"train_images={train_count}, val_images={val_count}"
        )
        return val_images, val_labels

    def _resolve_validation_dirs(
        self,
        train_images: Path,
        train_labels: Path,
        val_images: Optional[Path],
        val_labels: Optional[Path],
    ) -> Tuple[Path, Path]:
        val_ready = (
            val_images is not None
            and val_labels is not None
            and val_images.exists()
            and val_images.is_dir()
            and val_labels.exists()
            and val_labels.is_dir()
            and self._count_images(val_images) > 0
        )
        if val_ready:
            return val_images, val_labels

        if self.training_config.use_train_as_val_if_missing:
            logger.warning(
                "Validation dataset is missing or empty; using training data as validation. "
                "This is useful for smoke tests, but validation metrics will be optimistic."
            )
            return train_images, train_labels

        missing_paths = [
            self._format_optional_path(path, label)
            for path, label in [
                (val_images, "training.val_images"),
                (val_labels, "training.val_labels"),
            ]
            if path is None or not path.exists() or not path.is_dir()
        ]
        validation_path = (
            self._format_optional_path(val_images, "training.val_images")
            if val_images is None
            else str(val_images)
        )
        raise FileNotFoundError(
            "Validation dataset is missing or empty. "
            f"Missing/invalid paths: {missing_paths or [validation_path]}. "
            "Set training.use_train_as_val_if_missing=true to reuse training data."
        )

    @staticmethod
    def _required_path(path_value: object, label: str) -> Path:
        if path_value is None or (isinstance(path_value, str) and not path_value.strip()):
            raise ValueError(f"{label} is required when training.dataset_yaml is not set.")
        return Path(path_value)

    @staticmethod
    def _optional_path(path_value: object) -> Optional[Path]:
        if path_value is None or (isinstance(path_value, str) and not path_value.strip()):
            return None
        return Path(path_value)

    @staticmethod
    def _format_optional_path(path: Optional[Path], label: str) -> str:
        if path is None:
            return f"{label} is not configured"
        return str(path)

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
