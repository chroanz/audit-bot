FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl e usado pelo HEALTHCHECK e pelo entrypoint (espera o servidor subir
# antes de popular os dados de demonstracao).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY ml ./ml
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Gera o dataset sintetico e treina o modelo "auditado" de demonstracao
# em build-time. A geracao usa uma seed fixa (RNG(42)), entao o resultado
# e reprodutivel e nao depende de escrita em runtime.
RUN python ml/generate_and_train.py

RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=6 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
