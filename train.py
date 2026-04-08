"""Fine-tune YOLOv8-Seg for building instance segmentation."""
import argparse
from pathlib import Path
from typing import Optional

from src.core.training import YOLOSegFineTuner
from src.utils.config import PipelineConfig
from src.utils.logger import logger, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8-Seg on building segmentation PNG data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase1_yolov8.yaml",
        help="Pipeline YAML config path",
    )
    parser.add_argument(
        "--data-yaml",
        type=str,
        default=None,
        help="Existing Ultralytics dataset YAML. If omitted, one is generated from training dirs.",
    )
    parser.add_argument("--train-images", type=str, default=None, help="Training image directory")
    parser.add_argument("--val-images", type=str, default=None, help="Validation image directory")
    parser.add_argument("--train-labels", type=str, default=None, help="Training label directory")
    parser.add_argument("--val-labels", type=str, default=None, help="Validation label directory")
    parser.add_argument("--weights", type=str, default=None, help="Pretrained or previous checkpoint path")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Total batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Training image size")
    parser.add_argument("--workers", type=int, default=None, help="Data loader workers")
    parser.add_argument("--device", type=str, default=None, help="Ultralytics device string, e.g. 0,1,2,3 or cpu")
    parser.add_argument("--project", type=str, default=None, help="Training output project directory")
    parser.add_argument("--name", type=str, default=None, help="Training run name")
    parser.add_argument("--resume", action="store_true", help="Resume previous Ultralytics run")
    parser.add_argument(
        "--no-train-as-val-if-missing",
        action="store_true",
        help="Fail instead of using training data when validation data is missing",
    )
    return parser.parse_args()


def load_config(config_path: str) -> PipelineConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return PipelineConfig.from_yaml(str(path))


def apply_cli_overrides(config: PipelineConfig, args: argparse.Namespace) -> None:
    training = config.training
    training.enabled = True
    _set_if_not_none(training, "dataset_yaml", args.data_yaml)
    _set_if_not_none(training, "train_images", args.train_images)
    _set_if_not_none(training, "val_images", args.val_images)
    _set_if_not_none(training, "train_labels", args.train_labels)
    _set_if_not_none(training, "val_labels", args.val_labels)
    _set_if_not_none(training, "pretrained_weights", args.weights)
    _set_if_not_none(training, "epochs", args.epochs)
    _set_if_not_none(training, "batch", args.batch)
    _set_if_not_none(training, "imgsz", args.imgsz)
    _set_if_not_none(training, "workers", args.workers)
    _set_if_not_none(training, "device", args.device)
    _set_if_not_none(training, "project", args.project)
    _set_if_not_none(training, "name", args.name)
    if args.resume:
        training.resume = True
    if args.no_train_as_val_if_missing:
        training.use_train_as_val_if_missing = False


def _set_if_not_none(target: object, attr_name: str, value: Optional[object]) -> None:
    if value is not None:
        setattr(target, attr_name, value)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_cli_overrides(config, args)

    setup_logger(
        log_dir=str(config.data.output_dir / "logs"),
        log_level=config.log_level,
        name="training",
    )

    trainer = YOLOSegFineTuner(config)
    result = trainer.train()

    logger.info("Training output summary:")
    logger.info(f"  dataset_yaml: {result['dataset_yaml']}")
    logger.info(f"  run_dir: {result['run_dir']}")
    logger.info(f"  best_model: {result['best_model']}")
    logger.info(f"  last_model: {result['last_model']}")


if __name__ == "__main__":
    main()
