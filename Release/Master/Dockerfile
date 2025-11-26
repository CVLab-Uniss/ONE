FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV ADMIN_TOKEN="MySecureToken1234" \
    IOTHUB_CONNECTION_STRING="XXXXXXX" \
    K8S_TARGET_NAMESPACE="dev" \
    BASE_DOMAIN="example.nip.io" \
    DEFAULT_WORKLOAD_IMAGE="mfadd/one-project:wlchar_ad_api_cpu_v2" \
    DEFAULT_WORKLOAD_NAME="wlchar_ad_api_cpu_v2" \
    DEFAULT_LOG_LEVEL="INFO" \
    SLAVE_IMAGE="mfadd/slave-gateway:geo-v1" \
    SLAVE_CONTAINER_PORT="8080" \
    REGISTER_EDGE_ON_BOOTSTRAP="true" \
    HEALTHCHECK_TIMEOUT="25" \
    MONGO_URI="mongodb://mongo-service:27017/" \
    MONGO_DB_NAME="one_project_db"

EXPOSE 5000

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]

