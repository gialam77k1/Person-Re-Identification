ARG MLFLOW_BASE_IMAGE=ghcr.io/mlflow/mlflow:v3.10.0
FROM ${MLFLOW_BASE_IMAGE}

RUN python -m pip install --no-cache-dir \
    boto3==1.40.55 \
    psycopg2-binary==2.9.11
