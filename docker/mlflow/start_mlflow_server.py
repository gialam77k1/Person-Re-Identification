from __future__ import annotations

import os
from urllib.parse import quote_plus


def require_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    postgres_user = quote_plus(require_environment("POSTGRES_USER"))
    postgres_password = quote_plus(require_environment("POSTGRES_PASSWORD"))
    postgres_database = quote_plus(require_environment("POSTGRES_DB"))
    artifact_bucket = require_environment("MLFLOW_ARTIFACT_BUCKET")
    backend_store_uri = (
        f"postgresql+psycopg2://{postgres_user}:{postgres_password}"
        f"@postgres:5432/{postgres_database}"
    )
    os.execvp(
        "mlflow",
        [
            "mlflow",
            "server",
            "--host",
            "0.0.0.0",
            "--port",
            "5000",
            "--backend-store-uri",
            backend_store_uri,
            "--serve-artifacts",
            "--artifacts-destination",
            f"s3://{artifact_bucket}",
        ],
    )


if __name__ == "__main__":
    main()
