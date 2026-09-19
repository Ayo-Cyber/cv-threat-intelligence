from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from cvti.object_watch.images import decode_crop_encode, encode_rgb_array


def source_image() -> bytes:
    image = Image.new("RGB", (4, 2), "red")
    image.putpixel((3, 1), (0, 255, 0))
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def test_legacy_whole_image_and_explicit_pixel_crop_are_canonical_png():
    whole = decode_crop_encode(source_image(), (0, 0, 1, 1))
    crop = decode_crop_encode(source_image(), (2, 0, 4, 2), "pixel_xyxy")
    assert whole.pixel_bbox == (0, 0, 4, 2)
    assert Image.open(io.BytesIO(whole.png_bytes)).mode == "RGB"
    assert Image.open(io.BytesIO(crop.png_bytes)).size == (2, 2)


def test_normalized_crop_and_bounds_are_strict():
    result = decode_crop_encode(source_image(), (0.5, 0, 1, 1), "normalized_xyxy")
    assert result.pixel_bbox == (2, 0, 4, 2)
    for bbox, bbox_format in [
        ((0, 0, 2, 1), "normalized_xyxy"),
        ((-1, 0, 2, 1), "pixel_xyxy"),
        ((1, 1, 1, 2), "pixel_xyxy"),
        ((0, 0, float("nan"), 1), "pixel_xyxy"),
    ]:
        with pytest.raises(ValueError):
            decode_crop_encode(source_image(), bbox, bbox_format)


def test_exif_orientation_is_applied_before_crop():
    image = Image.new("RGB", (3, 2), "blue")
    exif = Image.Exif()
    exif[274] = 6
    output = io.BytesIO()
    image.save(output, "JPEG", exif=exif)
    result = decode_crop_encode(output.getvalue(), (0, 0, 1, 1))
    assert (result.source_width, result.source_height) == (2, 3)


def test_runtime_bgr_array_uses_same_decodable_rgb_contract():
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[:, :] = (255, 0, 0)
    decoded = Image.open(io.BytesIO(encode_rgb_array(bgr))).convert("RGB")
    assert decoded.getpixel((0, 0)) == (0, 0, 255)


def test_bad_image_bytes_are_rejected():
    with pytest.raises(ValueError, match="decodable"):
        decode_crop_encode(b"not an image", (0, 0, 1, 1))


@pytest.mark.parametrize("data", [
    b"\x89PNG\r\n\x1a\ncorrupt",
    b"\xff\xd8\xffcorrupt",
    b"RIFF\x04\x00\x00\x00WEBPcorrupt",
])
def test_corrupt_allowed_signature_is_rejected_without_public_image_open(monkeypatch, data):
    def forbidden(*args, **kwargs):
        raise AssertionError("globally patchable Image.open must not be called")

    monkeypatch.setattr(Image, "open", forbidden)
    with pytest.raises(ValueError, match="decodable"):
        decode_crop_encode(data, (0, 0, 1, 1))


@pytest.mark.parametrize("format_name", ["PNG", "JPEG"])
def test_valid_allowed_images_decode_when_public_image_open_is_patched(monkeypatch, format_name):
    image = Image.new("RGB", (3, 2), (12, 34, 56))
    output = io.BytesIO()
    image.save(output, format_name)
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("Image.open called"))
    result = decode_crop_encode(output.getvalue(), (0, 0, 1, 1))
    assert (result.source_width, result.source_height) == (3, 2)
