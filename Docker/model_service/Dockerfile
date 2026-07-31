FROM python:3.11

WORKDIR /code

RUN pip install rectools fastapi mlflow minio dotenv
RUN pip install boto3 psycopg2
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY src/backend /code/src/backend
COPY src/model /code/src/model
COPY ./als_model.npz /code/
COPY ./item_id_map.json /code/
COPY ./items_full_CLIP_wo_als.pickle /code/
COPY ./items_full_encoded_test.pickle /code/

CMD ["uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "8003"]