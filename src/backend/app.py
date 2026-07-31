from src.backend.tools import UserData, get_sql_engine, get_minio_client, get_recs
from scipy.sparse import csr_matrix
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
from io import BytesIO
from sqlalchemy import text
from implicit.als import AlternatingLeastSquares
import numpy as np
import pickle
import os
import mlflow


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

als_w = "models/als_model.npz"
iidm = "data/processed/item_id_map.json"
item_embs = "data/processed/items_full_CLIP_wo_als.pickle"
item_enc = "data/processed/items_full_encoded_test.pickle"
with open(iidm, "r") as jsfile: iid_map = json.load(jsfile)

model = Model("serve_test_user", als_w, iidm, item_embs, item_enc)

sql_engine = get_sql_engine()
minio_client = get_minio_client()

app = FastAPI()

@app.get("/get_item_ids")
async def get_item_ids():
    return list(iid_map.keys())

@app.get("/get_item_image")
async def get_item_image(item_id: str):
    image_stream = BytesIO(minio_client.get_object("images", item_id + ".png").read())
    return StreamingResponse(image_stream, media_type="application/octet-stream")

@app.get("/get_item_title")
async def get_item_name(item_id: str):
    conn = sql_engine.connect()
    query = text(f"""SELECT title FROM public.items WHERE parent_asin = '{item_id}'""")
    result = conn.execute(query).fetchall()[0][0]
    return result

@app.get("/get_popular_items")
async def get_popular_items(num_items: int):
    conn = sql_engine.connect()
    query = text(f"""SELECT parent_asin FROM public.items ORDER BY num_buys DESC LIMIT {num_items};""")
    result = conn.execute(query).fetchall()
    return list(map(lambda x: x[0], result))

@app.post("/recommend_with_model")
async def recommend_with_model(
    user_data: UserData, num_recs: int = 10
):
    recs = model.recommend(user_data, num_recs)
    return recs