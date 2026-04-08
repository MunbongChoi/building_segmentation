"""Convert TIF/GeoTIFF imagery to PNG files."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    from .logger import get_logger, setup_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    def setup_logger(log_level: str = "INFO", **_: object) -> None:
        logging.getLogger().setLevel(log_level)
else:
    logger = get_logger(__name__)


TIF_EXTENSIONS = {".tif", ".tiff", ".geotiff"}


def convert_tif_to_png(
    input_path: str,
    output_dir: str = "data",
    bands: Optional[Sequence[int]] = None,
    normalize: str = "auto",
    percentiles: Tuple[float, float] = (2.0, 98.0),
    tile_size: int = 0,
    overlap_ratio: float = 0.0,
    recursive: bool = False,
    overwrite: bool = False,
) -> List[Path]:
    """Convert a TIF file or directory of TIF files to PNG.

    Args:
        input_path: TIF file or directory containing TIF files.
        output_dir: Directory where PNG files will be written.
        bands: One band for grayscale or three 1-based bands for RGB/false color.
        normalize: auto, none, minmax, or percentile.
        percentiles: Lower/upper percentile for percentile normalization.
        tile_size: If > 0, write tiled PNGs of this size instead of one full PNG.
        overlap_ratio: Tile overlap ratio in [0, 1).
        recursive: Search input directories recursively.
        overwrite: Replace existing PNG files.

    Returns:
        List of written PNG paths.
    """
    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if normalize not in {"auto", "none", "minmax", "percentile"}:
        raise ValueError("normalize must be one of: auto, none, minmax, percentile")
    if tile_size < 0:
        raise ValueError("tile_size must be >= 0")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in the range [0, 1)")
    if percentiles[0] >= percentiles[1]:
        raise ValueError("percentiles must be ordered as lower < upper")

    tif_files = _collect_tif_files(source, recursive=recursive)
    if not tif_files:
        raise FileNotFoundError(f"No TIF files found: {source}")

    input_root = source if source.is_dir() else source.parent
    written_files: List[Path] = []
    for tif_file in tif_files:
        written_files.extend(
            convert_tif_file(
                tif_file=tif_file,
                output_dir=destination,
                input_root=input_root,
                bands=bands,
                normalize=normalize,
                percentiles=percentiles,
                tile_size=tile_size,
                overlap_ratio=overlap_ratio,
                overwrite=overwrite,
            )
        )

    logger.info(f"Converted {len(tif_files)} TIF file(s) into {len(written_files)} PNG file(s)")
    return written_files


def convert_tif_file(
    tif_file: Path,
    output_dir: Path,
    input_root: Path,
    bands: Optional[Sequence[int]] = None,
    normalize: str = "auto",
    percentiles: Tuple[float, float] = (2.0, 98.0),
    tile_size: int = 0,
    overlap_ratio: float = 0.0,
    overwrite: bool = False,
) -> List[Path]:
    """Convert one TIF file to one or more PNG files."""
    import rasterio
    from rasterio.windows import Window

    logger.info(f"Converting TIF: {tif_file}")
    output_base = _output_base_for(tif_file, input_root=input_root, output_dir=output_dir)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_file) as src:
        band_indexes = _resolve_band_indexes(src.count, bands)
        band_dtypes = [src.dtypes[index - 1] for index in band_indexes]

        if tile_size <= 0:
            output_file = output_base.with_suffix(".png")
            if output_file.exists() and not overwrite:
                logger.info(f"Skipping existing PNG: {output_file}")
                return []
            rgb = _read_window_as_uint8_rgb(
                src=src,
                window=None,
                band_indexes=band_indexes,
                band_dtypes=band_dtypes,
                normalize=normalize,
                percentiles=percentiles,
            )
            _write_png(output_file, rgb)
            return [output_file]

        stride = max(1, int(tile_size * (1 - overlap_ratio)))
        written_files = []
        for y in _axis_offsets(src.height, tile_size, stride):
            for x in _axis_offsets(src.width, tile_size, stride):
                window_width = min(tile_size, src.width - x)
                window_height = min(tile_size, src.height - y)
                window = Window(x, y, window_width, window_height)
                output_file = output_base.parent / (
                    f"{output_base.name}_x{int(x):06d}_y{int(y):06d}.png"
                )
                if output_file.exists() and not overwrite:
                    logger.info(f"Skipping existing PNG: {output_file}")
                    continue

                rgb = _read_window_as_uint8_rgb(
                    src=src,
                    window=window,
                    band_indexes=band_indexes,
                    band_dtypes=band_dtypes,
                    normalize=normalize,
                    percentiles=percentiles,
                )
                _write_png(output_file, rgb)
                written_files.append(output_file)

        return written_files


def _collect_tif_files(source: Path, recursive: bool) -> List[Path]:
    if source.is_file():
        if source.suffix.lower() not in TIF_EXTENSIONS:
            raise ValueError(f"Input file is not a supported TIF: {source}")
        return [source]

    if not source.exists():
        raise FileNotFoundError(f"Input path not found: {source}")
    if not source.is_dir():
        raise ValueError(f"Input path must be a file or directory: {source}")

    iterator: Iterable[Path] = source.rglob("*") if recursive else source.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in TIF_EXTENSIONS
    )


def _resolve_band_indexes(source_band_count: int, bands: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if source_band_count <= 0:
        raise ValueError("Source raster has no bands")

    if bands is None:
        return (1, 2, 3) if source_band_count >= 3 else (1,)

    resolved = tuple(int(index) for index in bands)
    if len(resolved) not in {1, 3}:
        raise ValueError("bands must contain either one band or three bands")
    invalid = [index for index in resolved if index < 1 or index > source_band_count]
    if invalid:
        raise ValueError(
            f"Band indexes out of range: {invalid}. Source has {source_band_count} band(s)."
        )
    return resolved


def _read_window_as_uint8_rgb(
    src,
    window: Optional[Window],
    band_indexes: Sequence[int],
    band_dtypes: Sequence[str],
    normalize: str,
    percentiles: Tuple[float, float],
) -> np.ndarray:
    import numpy as np

    image = src.read(list(band_indexes), window=window, masked=True).astype(np.float32)
    image = np.ma.filled(image, np.nan)

    channels = []
    for channel_index, dtype_name in enumerate(band_dtypes):
        channels.append(
            _scale_band_to_uint8(
                image[channel_index],
                dtype_name=dtype_name,
                normalize=normalize,
                percentiles=percentiles,
            )
        )

    rgb = np.stack(channels, axis=-1)
    if rgb.shape[-1] == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    return rgb


def _scale_band_to_uint8(
    band: np.ndarray,
    dtype_name: str,
    normalize: str,
    percentiles: Tuple[float, float],
) -> np.ndarray:
    import numpy as np

    method = normalize
    if method == "auto":
        method = "none" if dtype_name == "uint8" else "percentile"

    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)

    if method == "none":
        scaled = np.clip(band, 0, 255)
    elif method == "minmax":
        lower = float(np.min(valid))
        upper = float(np.max(valid))
        scaled = _linear_scale(band, lower, upper)
    elif method == "percentile":
        lower, upper = np.nanpercentile(valid, percentiles)
        scaled = _linear_scale(band, float(lower), float(upper))
    else:
        raise ValueError(f"Unsupported normalization method: {normalize}")

    return np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)


def _linear_scale(band: np.ndarray, lower: float, upper: float) -> np.ndarray:
    import numpy as np

    if upper <= lower:
        return np.zeros(band.shape, dtype=np.float32)
    return np.clip((band - lower) * 255.0 / (upper - lower), 0, 255)


def _write_png(output_file: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    output_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_file)
    logger.info(f"Wrote PNG: {output_file}")


def _output_base_for(tif_file: Path, input_root: Path, output_dir: Path) -> Path:
    try:
        relative_path = tif_file.relative_to(input_root)
    except ValueError:
        relative_path = Path(tif_file.name)
    return (output_dir / relative_path).with_suffix("")


def _axis_offsets(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]

    offsets = list(range(0, length - tile_size + 1, stride))
    last_offset = length - tile_size
    if offsets[-1] != last_offset:
        offsets.append(last_offset)
    return offsets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert TIF/GeoTIFF files to PNG")
    parser.add_argument("--input", required=True, help="TIF file or directory")
    parser.add_argument("--output-dir", default="data", help="PNG output directory")
    parser.add_argument(
        "--bands",
        type=int,
        nargs="+",
        default=None,
        help="One grayscale band or three RGB/false-color bands. Example: --bands 4 3 2",
    )
    parser.add_argument(
        "--normalize",
        choices=["auto", "none", "minmax", "percentile"],
        default="auto",
        help="8-bit scaling method",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        default=(2.0, 98.0),
        metavar=("LOW", "HIGH"),
        help="Percentile stretch range used by --normalize percentile/auto",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="If > 0, split each TIF into tiled PNGs of this size",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.0,
        help="Tile overlap ratio when --tile-size is used",
    )
    parser.add_argument("--recursive", action="store_true", help="Search input directories recursively")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logger(log_level="INFO")
    convert_tif_to_png(
        input_path=args.input,
        output_dir=args.output_dir,
        bands=args.bands,
        normalize=args.normalize,
        percentiles=tuple(args.percentiles),
        tile_size=args.tile_size,
        overlap_ratio=args.overlap_ratio,
        recursive=args.recursive,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
