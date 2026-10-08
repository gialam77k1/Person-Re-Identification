from __future__ import annotations

import os

import boto3


def main() -> None:
    required_variables = (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MLFLOW_S3_ENDPOINT_URL",
        "MLFLOW_ARTIFACT_BUCKET",
        "DVC_BUCKET",
    )
    missing = [name for name in required_variables if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    existing_buckets = {bucket["Name"] for bucket in client.list_buckets()["Buckets"]}
    for bucket_name in (
        os.environ["MLFLOW_ARTIFACT_BUCKET"],
        os.environ["DVC_BUCKET"],
    ):
        if bucket_name not in existing_buckets:
            client.create_bucket(Bucket=bucket_name)
            print(f"Created private bucket: {bucket_name}")
        else:
            print(f"Private bucket already exists: {bucket_name}")


if __name__ == "__main__":
    main()
