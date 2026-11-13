from __future__ import annotations

import re
from pathlib import Path


_IMAGE_FOLDER_PATTERN = re.compile(
    r"^(?P<pid>-?\d+)_c(?P<camera>\d+)(?:s(?P<sequence>\d+))?(?:_|$)",
    re.IGNORECASE,
)


def parse_image_folder_reid_filename(path_like: str | Path) -> tuple[int, int]:
    """Return ``(person_id, zero_based_camera_id)`` for common ReID names.

    DukeMTMC-reID uses names such as ``0002_c1_f...jpg`` while Market-1501
    adds a sequence to the camera token, for example
    ``0002_c1s1_000451_03.jpg``.  The sequence is capture metadata, not part of
    the camera id.  Keeping this parser dataset-neutral lets every iterative
    release use the same validated filename contract.
    """

    name = Path(path_like).name
    match = _IMAGE_FOLDER_PATTERN.match(Path(name).stem)
    if match is None:
        raise ValueError(
            "Invalid image-folder ReID filename: "
            f"{name}. Expected '<person_id>_c<camera>_...' or "
            "'<person_id>_c<camera>s<sequence>_...'."
        )

    camera = int(match.group("camera"))
    if camera < 1:
        raise ValueError(f"ReID camera ids must be one-based in filename: {name}")
    return int(match.group("pid")), camera - 1
