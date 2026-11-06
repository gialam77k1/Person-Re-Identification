FROM python:3.11.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements-control.txt /app/requirements-control.txt
RUN pip install --no-cache-dir -r /app/requirements-control.txt
COPY src /app/src
COPY configs/retraining_decision.yaml /app/configs/retraining_decision.yaml

EXPOSE 8090
CMD ["uvicorn", "src.control_api.main:app", "--host", "0.0.0.0", "--port", "8090"]
