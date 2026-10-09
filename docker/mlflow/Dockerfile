ARG MLFLOW_BASE_IMAGE=ghcr.io/mlflow/mlflow:v3.10.0
FROM ${MLFLOW_BASE_IMAGE}

RUN python -m pip install --no-cache-dir \
    boto3==1.40.55 \
    psycopg2-binary==2.9.11

COPY docker/mlflow/init_minio_buckets.py /opt/reid/init_minio_buckets.py
COPY docker/mlflow/start_mlflow_server.py /opt/reid/start_mlflow_server.py
