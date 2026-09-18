"""Canonical, bounded image handling shared by enrollment and recognition."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Any, Sequence


IMAGE_PREPROCESSING_VERSION = 1
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_DIMENSION = 16_384


def _decode_supported_image(data: bytes):
    """Decode an allow-listed format without PIL's globally patchable dispatcher."""
    from PIL import JpegImagePlugin, PngImagePlugin, WebPImagePlugin

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        decoder = PngImagePlugin.PngImageFile
    elif data.startswith(b"\xff\xd8\xff"):
        decoder = JpegImagePlugin.JpegImageFile
    elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        decoder = WebPImagePlugin.WebPImageFile
    else:
        raise ValueError("image format must be PNG, JPEG, or WEBP")

    # Plugin constructors call their own format parser directly. In particular,
    # this cannot enter Ultralytics' Image.open fallback/check_requirements path.
    return decoder(io.BytesIO(data))


@dataclass(frozen=True)
class CanonicalImage:
    png_bytes: bytes
    pixel_bbox: tuple[int, int, int, int]
    source_sha256: str
    source_width: int
    source_height: int
    preprocessing_version: int = IMAGE_PREPROCESSING_VERSION


def decode_crop_encode(
    image_bytes: bytes,
    bbox: Sequence[float | int],
    bbox_format: str = "legacy",
) -> CanonicalImage:
    """Decode with EXIF orientation, crop, and encode deterministic RGB PNG.

    Legacy ``[0, 0, 1, 1]`` means the entire oriented image; every other legacy
    box is interpreted as pixel XYXY. New callers should specify
    ``pixel_xyxy`` or ``normalized_xyxy`` explicitly.
    """
    data = bytes(image_bytes)
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image is empty or exceeds the byte limit")
    try:
        from PIL import ImageOps

        with _decode_supported_image(data) as opened:
            _validate_dimensions(*opened.size)
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:
        raise ValueError("image bytes are not a supported decodable image") from exc
    width, height = image.size
    _validate_dimensions(width, height)
    pixel_bbox = _resolve_bbox(bbox, bbox_format, width, height)
    crop = image.crop(pixel_bbox)
    output = io.BytesIO()
    crop.save(output, format="PNG", optimize=False, compress_level=6)
    return CanonicalImage(
        png_bytes=output.getvalue(),
        pixel_bbox=pixel_bbox,
        source_sha256=hashlib.sha256(data).hexdigest(),
        source_width=width,
        source_height=height,
    )


def encode_rgb_array(image: Any, *, colour_space: str = "bgr") -> bytes:
    """Encode an HxWx3 uint8 array as the same canonical RGB PNG contract."""
    import numpy as np
    from PIL import Image

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("image array must be HxWx3 uint8")
    _validate_dimensions(int(array.shape[1]), int(array.shape[0]))
    if colour_space == "bgr":
        array = array[:, :, ::-1]
    elif colour_space != "rgb":
        raise ValueError("colour_space must be 'bgr' or 'rgb'")
    output = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(array)).save(
        output, format="PNG", optimize=False, compress_level=6
    )
    return output.getvalue()


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0 or width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError("image dimensions are invalid or exceed the limit")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("decoded image exceeds the pixel limit")


def _resolve_bbox(
    bbox: Sequence[float | int], bbox_format: str, width: int, height: int
) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError("bbox must have four coordinates")
    values = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox coordinates must be finite")
    if bbox_format == "legacy":
        if values == (0.0, 0.0, 1.0, 1.0):
            return (0, 0, width, height)
        bbox_format = "pixel_xyxy"
    if bbox_format == "normalized_xyxy":
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("normalized bbox coordinates must be between 0 and 1")
        x1, y1, x2, y2 = (
            values[0] * width, values[1] * height,
            values[2] * width, values[3] * height,
        )
    elif bbox_format == "pixel_xyxy":
        x1, y1, x2, y2 = values
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError("pixel bbox must be within the oriented image")
    else:
        raise ValueError("bbox_format must be legacy, pixel_xyxy, or normalized_xyxy")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive area")
    # Floor starts and ceil ends so a valid normalized crop does not disappear.
    result = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("bbox must have positive pixel area")
    return result
