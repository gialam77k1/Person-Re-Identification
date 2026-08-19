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
    parser.add_argument("--max-batch-size", type=int, default=0)
    parser.add_argument("--input-height", type=int, default=224)
    parser.add_argument("--input-width", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--preferred-batch-sizes", default="4,8,16")
    parser.add_argument("--instance-kind", default="KIND_CPU")
    return parser.parse_args()


def build_config_pbtxt(
    model_name: str,
    max_batch_size: int,
    input_height: int,
    input_width: int,
    embedding_dim: int,
    preferred_batch_sizes: list[int],
    instance_kind: str,
) -> str:
    preferred_batch_line = ", ".join(str(size) for size in preferred_batch_sizes if size > 0)
    input_dims = f"[ 3, {input_height}, {input_width} ]" if max_batch_size > 0 else f"[ 1, 3, {input_height}, {input_width} ]"
    output_dims = f"[ {embedding_dim} ]" if max_batch_size > 0 else f"[ 1, {embedding_dim} ]"
    dynamic_batching_block = (
        f"""
dynamic_batching {{
  preferred_batch_size: [ {preferred_batch_line} ]
  max_queue_delay_microseconds: 2000
}}
"""
        if max_batch_size > 0 and preferred_batch_line
        else ""
    )

    return f"""name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}
input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: {input_dims}
  }}
]
output [
  {{
    name: "embeddings"
    data_type: TYPE_FP32
    dims: {output_dims}
  }}
]
{dynamic_batching_block}instance_group [
  {{
    kind: {instance_kind}
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
        external_data_path = onnx_path.with_name(f"{onnx_path.name}.data")
        copied_external_data_path = None
        if external_data_path.exists():
            copied_external_data_path = version_root / external_data_path.name
            shutil.copy2(external_data_path, copied_external_data_path)

        config_pbtxt = build_config_pbtxt(
            model_name=args.model_name,
            max_batch_size=args.max_batch_size,
            input_height=args.input_height,
            input_width=args.input_width,
            embedding_dim=args.embedding_dim,
            preferred_batch_sizes=preferred_batch_sizes,
            instance_kind=args.instance_kind,
        )
        (model_root / "config.pbtxt").write_text(config_pbtxt, encoding="utf-8")

        manifest = {
            "model_name": args.model_name,
            "model_version": args.model_version,
            "source_onnx_path": str(onnx_path),
            "triton_repository_root": str(output_root.resolve()),
            "triton_model_root": str(model_root.resolve()),
            "copied_onnx_path": str(target_onnx_path.resolve()),
            "copied_external_data_path": (
                str(copied_external_data_path.resolve()) if copied_external_data_path is not None else None
            ),
            "max_batch_size": args.max_batch_size,
            "input_shape": [3, args.input_height, args.input_width],
            "output_name": "embeddings",
            "embedding_dim": args.embedding_dim,
            "preferred_batch_sizes": preferred_batch_sizes,
            "instance_kind": args.instance_kind,
        }
        save_json(manifest, output_root / "triton_model_manifest.json")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
