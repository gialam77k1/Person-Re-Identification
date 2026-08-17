from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.utils import ensure_dir, resolve_path, save_json, tee_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-name", default="reid_embedding")
    parser.add_argument("--model-version", default="1")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--input-height", type=int, default=224)
    parser.add_argument("--input-width", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--preferred-batch-sizes", default="4,8,16")
    return parser.parse_args()


def build_config_pbtxt(
    model_name: str,
    max_batch_size: int,
    input_height: int,
    input_width: int,
    embedding_dim: int,
    preferred_batch_sizes: list[int],
) -> str:
    preferred_batch_line = ", ".join(str(size) for size in preferred_batch_sizes if size > 0)
    return f"""name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}
input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, {input_height}, {input_width} ]
  }}
]
output [
  {{
    name: "embeddings"
    data_type: TYPE_FP32
    dims: [ {embedding_dim} ]
  }}
]
dynamic_batching {{
  preferred_batch_size: [ {preferred_batch_line} ]
  max_queue_delay_microseconds: 2000
}}
instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]
"""


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    log_path = output_root / "prepare_triton_model.log"

    with tee_output(log_path):
        onnx_path = resolve_path(args.onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        preferred_batch_sizes = [
            int(raw.strip()) for raw in args.preferred_batch_sizes.split(",") if raw.strip()
        ]
        model_root = output_root / args.model_name
        version_root = model_root / args.model_version
        ensure_dir(version_root)

        target_onnx_path = version_root / "model.onnx"
        shutil.copy2(onnx_path, target_onnx_path)

        config_pbtxt = build_config_pbtxt(
            model_name=args.model_name,
            max_batch_size=args.max_batch_size,
            input_height=args.input_height,
            input_width=args.input_width,
            embedding_dim=args.embedding_dim,
            preferred_batch_sizes=preferred_batch_sizes,
        )
        (model_root / "config.pbtxt").write_text(config_pbtxt, encoding="utf-8")

        manifest = {
            "model_name": args.model_name,
            "model_version": args.model_version,
            "source_onnx_path": str(onnx_path),
            "triton_repository_root": str(output_root.resolve()),
            "triton_model_root": str(model_root.resolve()),
            "copied_onnx_path": str(target_onnx_path.resolve()),
            "max_batch_size": args.max_batch_size,
            "input_shape": [3, args.input_height, args.input_width],
            "output_name": "embeddings",
            "embedding_dim": args.embedding_dim,
            "preferred_batch_sizes": preferred_batch_sizes,
        }
        save_json(manifest, output_root / "triton_model_manifest.json")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
