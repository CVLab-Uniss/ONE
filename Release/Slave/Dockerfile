# Base leggera Python 3.9 slim
FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copia codice e requirements
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt

# Install deps di sistema minime (ca-certificates giusto per HTTPS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Lib Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Variabili d'ambiente di default (possono essere override nel deployment)
ENV PORT=8080 \
    ZONE="unknown" \
    LOG_LEVEL="INFO"

# Espongo la porta http del servizio
EXPOSE 8080

# Avvio con gunicorn (prod-ready)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "app:app"]
