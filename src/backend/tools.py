from implicit.als import AlternatingLeastSquares
import mlflow
import numpy as np
import json
from scipy.sparse import csr_matrix
import pickle
import torch
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import typing as tp
from minio import Minio
from sqlalchemy import create_engine


load_dotenv()

def get_recs(
        user_embedding: np.ndarray,
        items_embs: np.ndarray,
        n_recs: int,
        its: list
) -> list:

    scores = (user_embedding @ items_embs.T)
    items_rec = scores.argsort()[:, ::-1][:, :n_recs].squeeze()
    items_for_rec = []
    for item in items_rec:
        items_for_rec.append(its[item])

    return items_for_rec

def get_sql_engine():
    connection_string = f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('DATABASE_HOST')}/{os.getenv('POSTGRES_DB')}"
    engine = create_engine(connection_string)
    return engine

def get_minio_client():
    minio_client = Minio(
        os.getenv("MINIO_HOST"),
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        secure=False
    )
    return minio_client


class UserData(BaseModel):
    items: tp.List[str]
    ranks: tp.List[int]


class Model:

    def __init__(self, model_name: str, als_weights: str, item_id_map: str, item_embeddings_path: str, item_encoded_path: str):

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        self.als_model = AlternatingLeastSquares()
        self.als_model = self.als_model.load(als_weights)
        self.user_encoder = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        with open(item_id_map, "r") as jsfile: self.iid_map = json.load(jsfile)
        with open(item_embeddings_path, "rb") as f: self.item_embeddings = pickle.load(f)
        with open(item_encoded_path, "rb") as f: self.item_encoded = pickle.load(f)
    
    def recommend(self, user_data: UserData, n_recs: int):
        items, ranks = user_data.items, user_data.ranks

        user_collaborative_emb = np.zeros((len(self.iid_map), ))
        for item in items:
            user_collaborative_emb[self.iid_map[item]] = 1
        user_collaborative_emb = csr_matrix(user_collaborative_emb)
        user_als_emb = self.als_model.recalculate_user(0, user_collaborative_emb)

        user_content_emb = 0
        for i, item in enumerate(items):
            user_content_emb += ranks[i] * self.item_embeddings[item]
        user_content_emb = user_content_emb / np.sum(ranks)

        user_emb = self.user_encoder.predict(np.concatenate((user_content_emb, user_als_emb), dtype=np.float32).reshape((-1, 2615)))

        recs = get_recs(
            user_emb,
            np.array(list(self.item_encoded.values())),
            n_recs,
            list(self.item_encoded.keys())
            )

        return recs

