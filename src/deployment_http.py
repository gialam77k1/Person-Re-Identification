from __future__ import annotations

import json
import mimetypes
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def get_json(url: str, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc}") from exc


def post_image_bytes(
    url: str,
    image: bytes,
    *,
    filename: str,
    content_type: str,
    query: dict[str, str | int] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    boundary = f"----reid-{uuid.uuid4().hex}"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        url,
        data=prefix + image + suffix,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc}") from exc


def post_image_path(
    url: str,
    image_path: str | Path,
    *,
    query: dict[str, str | int] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    path = Path(image_path)
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return post_image_bytes(
        url,
        path.read_bytes(),
        filename=path.name,
        content_type=content_type,
        query=query,
        timeout=timeout,
    )
