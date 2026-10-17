from __future__ import annotations

import io
import warnings
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


ALLOWED_CONTENT_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG"}


class ImageValidationError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def validate_and_decode_image(
    raw: bytes,
    content_type: str | None,
    *,
    max_upload_bytes: int,
    max_image_pixels: int,
) -> Image.Image:
    if not raw:
        raise ImageValidationError("Image file is empty.")
    if len(raw) > max_upload_bytes:
        raise ImageValidationError(
            f"Image exceeds the {max_upload_bytes}-byte upload limit.",
            status_code=413,
        )
    expected_format = ALLOWED_CONTENT_TYPES.get((content_type or "").lower())
    if expected_format is None:
        raise ImageValidationError("Only image/jpeg and image/png are accepted.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            candidate = Image.open(io.BytesIO(raw))
            detected_format = candidate.format
            width, height = candidate.size
            candidate.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError("The uploaded file is not a valid JPG or PNG image.") from exc

    if detected_format != expected_format:
        raise ImageValidationError(
            f"Content-Type declares {expected_format}, but the image is {detected_format or 'unknown'}."
        )
    if width <= 0 or height <= 0 or width * height > max_image_pixels:
        raise ImageValidationError(
            f"Image dimensions exceed the {max_image_pixels}-pixel limit.",
            status_code=413,
        )

    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError("The uploaded image could not be decoded.") from exc


def preprocess_image(image: Image.Image, contract: dict[str, Any]) -> np.ndarray:
    resize = contract.get("resize", {})
    padding = contract.get("padding", {})
    normalization = contract.get("normalization", {})
    conversion = contract.get("tensor_conversion", {})
    target_height = int(resize.get("target_height", 0))
    target_width = int(resize.get("target_width", 0))
    if target_height <= 0 or target_width <= 0:
        raise ValueError("Preprocessing contract has invalid target dimensions.")
    if resize.get("preserve_aspect_ratio") is not True:
        raise ValueError("Only preserve-aspect-ratio preprocessing is supported.")
    if resize.get("interpolation") != "bilinear":
        raise ValueError("Only bilinear preprocessing is supported.")

    source_width, source_height = image.size
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = min(target_width, max(1, round(source_width * scale)))
    resized_height = min(target_height, max(1, round(source_height * scale)))
    resized = image.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    fill = tuple(int(value) for value in padding.get("fill_rgb", []))
    if len(fill) != 3:
        raise ValueError("Preprocessing contract must define three RGB padding values.")
    canvas = Image.new("RGB", (target_width, target_height), color=fill)
    canvas.paste(
        resized,
        ((target_width - resized_width) // 2, (target_height - resized_height) // 2),
    )

    mean = np.asarray(normalization.get("mean"), dtype=np.float32)
    std = np.asarray(normalization.get("std"), dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std == 0):
        raise ValueError("Preprocessing normalization must contain valid RGB mean/std.")
    pixel_scale = float(conversion.get("pixel_scale", 1.0 / 255.0))
    if not np.isfinite(pixel_scale) or pixel_scale <= 0:
        raise ValueError("Preprocessing pixel_scale must be a positive finite number.")
    tensor = np.asarray(canvas, dtype=np.float32) / (1.0 / pixel_scale)
    tensor = (tensor - mean) / std
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return tensor.astype(np.float32)
