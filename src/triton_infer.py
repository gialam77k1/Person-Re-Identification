from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.utils import ensure_dir, save_json, tee_output


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model-name", default="reid_embedding")
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--output-name", default="embeddings")
    parser.add_argument("--input-height", type=int, default=224)
    parser.add_argument("--input-width", type=int, default=224)
    parser.add_argument("--output-root", default="artifacts/inference/local-triton")
    return parser.parse_args()


def preprocess_image(image_path: str | Path, input_height: int, input_width: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((input_width, input_height))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    return image_array.astype(np.float32)


def call_triton_http(
    server_url: str,
    model_name: str,
    input_name: str,
    output_name: str,
    tensor: np.ndarray,
) -> dict:
    payload = {
        "inputs": [
            {
                "name": input_name,
                "shape": list(tensor.shape),
                "datatype": "FP32",
                "data": tensor.reshape(-1).tolist(),
            }
        ],
        "outputs": [
            {
                "name": output_name,
            }
        ],
    }
    request_body = json.dumps(payload).encode("utf-8")
    endpoint = f"{server_url.rstrip('/')}/v2/models/{model_name}/infer"
    request = urllib.request.Request(
        endpoint,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_triton_output(response: dict) -> np.ndarray:
    outputs = response.get("outputs", [])
    if not outputs:
        raise ValueError("Triton response does not contain outputs.")

    output_spec = outputs[0]
    output_shape = output_spec.get("shape")
    output_data = output_spec.get("data")
    if output_data is None:
        raise ValueError(
            "Triton response does not contain inline output data. "
            "This client currently expects JSON output, not binary output."
        )

    embedding = np.asarray(output_data, dtype=np.float32)
    if output_shape:
        embedding = embedding.reshape(output_shape)
    return embedding


def infer_embedding(
    server_url: str,
    model_name: str,
    input_name: str,
    output_name: str,
    tensor: np.ndarray,
) -> np.ndarray:
    response = call_triton_http(
        server_url=server_url,
        model_name=model_name,
        input_name=input_name,
        output_name=output_name,
        tensor=tensor,
    )
    return parse_triton_output(response)


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    log_path = output_root / "triton_infer.log"

    with tee_output(log_path):
        print(f"Logging console output to {log_path}")
        tensor = preprocess_image(args.image_path, args.input_height, args.input_width)
        print(f"Prepared input tensor with shape {tuple(tensor.shape)}")

        try:
            embedding = infer_embedding(
                server_url=args.server_url,
                model_name=args.model_name,
                input_name=args.input_name,
                output_name=args.output_name,
                tensor=tensor,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Triton HTTP error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Triton server at {args.server_url}. "
                "Ensure docker compose is running and the HTTP port is exposed."
            ) from exc

        image_stem = Path(args.image_path).stem
        embedding_path = output_root / f"{image_stem}_embedding.npy"
        np.save(embedding_path, embedding)

        manifest = {
            "image_path": str(Path(args.image_path).resolve()),
            "server_url": args.server_url,
            "model_name": args.model_name,
            "input_name": args.input_name,
            "output_name": args.output_name,
            "input_shape": list(tensor.shape),
            "output_shape": list(embedding.shape),
            "embedding_path": str(embedding_path.resolve()),
            "embedding_preview": embedding.reshape(-1)[:10].tolist(),
        }
        save_json(manifest, output_root / f"{image_stem}_infer_manifest.json")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
