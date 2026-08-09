FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

RUN pip install --no-cache-dir fastapi uvicorn mlflow minio python-dotenv catboost tqdm boto3

RUN pip install --no-cache-dir "feast[postgres]" redis qdrant-client psycopg2-binary

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

COPY src/backend /code/src/backend
COPY src/model /code/src/model

COPY feature_repo /code/feature_repo

COPY data/processed/item_id_map.json /code/

CMD ["uvicorn", "src.backend.app2:app", "--host", "0.0.0.0", "--port", "8003"]