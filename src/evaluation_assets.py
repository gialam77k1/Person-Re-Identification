from __future__ import annotations

import base64
import binascii
import hashlib
from typing import Any, Iterator


ALLOWED_PREVIEW_MIME_TYPES = {"image/jpeg", "image/png"}


def preview_images(report: Any) -> Iterator[dict[str, Any]]:
    if not isinstance(report, dict):
        return
    preview = report.get("retrieval_preview")
    if not isinstance(preview, list):
        return
    for item in preview:
        if not isinstance(item, dict):
            continue
        query = item.get("query")
        if isinstance(query, dict):
            yield query
        predictions = item.get("predictions")
        if isinstance(predictions, list):
            for prediction in predictions:
                if isinstance(prediction, dict):
                    yield prediction


def decode_preview_asset(image: dict[str, Any]) -> tuple[bytes, str]:
    asset_id = str(image.get("asset_id", ""))
    mime = str(image.get("asset_mime", ""))
    encoded = image.get("asset_data")
    if len(asset_id) != 64 or any(char not in "0123456789abcdef" for char in asset_id):
        raise ValueError("Evaluation preview asset_id is missing or invalid.")
    if mime not in ALLOWED_PREVIEW_MIME_TYPES:
        raise ValueError("Evaluation preview asset MIME type is not allowed.")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("Evaluation preview asset_data is missing.")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Evaluation preview asset_data is not valid base64.") from exc
    if not content or hashlib.sha256(content).hexdigest() != asset_id:
        raise ValueError("Evaluation preview asset checksum does not match asset_id.")
    signatures = {
        "image/jpeg": (b"\xff\xd8\xff",),
        "image/png": (b"\x89PNG\r\n\x1a\n",),
    }
    if not any(content.startswith(signature) for signature in signatures[mime]):
        raise ValueError("Evaluation preview asset content does not match its MIME type.")
    return content, mime


def validate_preview_assets(report: dict[str, Any]) -> int:
    preview = report.get("retrieval_preview")
    if not isinstance(preview, list) or not preview:
        raise ValueError("Evaluation retrieval_preview is missing or empty.")
    count = 0
    for image in preview_images(report):
        decode_preview_asset(image)
        count += 1
    if count == 0:
        raise ValueError("Evaluation retrieval_preview contains no images.")
    return count


def find_preview_asset(report: Any, asset_id: str) -> tuple[bytes, str] | None:
    for image in preview_images(report):
        if image.get("asset_id") == asset_id:
            return decode_preview_asset(image)
    return None
