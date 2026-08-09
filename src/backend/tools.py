import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import typing as tp
from minio import Minio
from sqlalchemy import create_engine
from src.model.utils import ItemFeatureProjector, UserTower
from catboost import Pool, CatBoostRanker
from qdrant_client import QdrantClient
from feast import FeatureStore
import json


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

class UserDataT(BaseModel):
    items: tp.List[str]

def load_transformer_model():
    device = torch.device("cuda")
    run_id = "31140806f171423792c3e72730a93578"
    artifact_uri = f"runs:/{run_id}/model_weights/fps_loss_transformer_user_test_1.pth"
    local_weight_path = mlflow.artifacts.download_artifacts(
        artifact_uri = artifact_uri,
        dst_path="/tmp/models"
    )
    image_text_dim = 512
    als_dim = 100
    cat_dim = 2615 - 2*image_text_dim - als_dim
    proj_m = ItemFeatureProjector(als_dim, image_text_dim, image_text_dim, cat_dim)
    user_encoder = UserTower(proj_m)
    user_weights = torch.load(local_weight_path)
    user_encoder.load_state_dict(user_weights)
    user_encoder.eval()
    user_encoder.to(device)
    return user_encoder

def load_catboost_model():
    run_id = "8f7fddd1cc904517947f6f79714f0ac1"
    artifact_uri = f"runs:/{run_id}/model_weights/boosting_boosting2.cbm"
    local_weight_path = mlflow.artifacts.download_artifacts(
        artifact_uri = artifact_uri,
        dst_path="/tmp/models"
    )
    boosting = CatBoostRanker()
    boosting.load_model(local_weight_path)
    return boosting

class ModelTransformer():
    def __init__(self):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.user_encoder = load_transformer_model()
        self.boosting = load_catboost_model()
        
        self.qdrant_client = QdrantClient(host="qdrant", port=6334, prefer_grpc=True)
        self.feast_store = FeatureStore(repo_path="./feature_repo")

    def recommend(self, user_data: list, n_recs: int):
        user_history = user_data.items
        real_history_ids = [str(item) for item in user_history if item != "0"]

        history_entities = [{"candidate_item_id": item_id} for item_id in real_history_ids]
        history_features = self.feast_store.get_online_features(
            features=["item_features:transformer_input_vector"], 
            entity_rows=history_entities
        ).to_dict()
        
        fetched_transformer_vectors = {}
        for item_id, vec in zip(history_features["candidate_item_id"], history_features["transformer_input_vector"]):
            if vec and vec != "None":
                try:
                    fetched_transformer_vectors[item_id] = json.loads(vec)
                except Exception:
                    fetched_transformer_vectors[item_id] = np.zeros((128,), dtype=np.float32).tolist()
        
        seq_vect_inf = []
        for item in user_history:
            item_str = str(item)
            if item_str != "0" and item_str in fetched_transformer_vectors:
                seq_vect_inf.append(fetched_transformer_vectors[item_str])
            else:
                seq_vect_inf.append(np.zeros((128,), dtype=np.float32).tolist())  
        seq_vect_inf = torch.from_numpy(np.asarray(seq_vect_inf, dtype=np.float32).reshape(1, 10, -1)).to(self.device)
        
        with torch.no_grad():
            user_vec = self.user_encoder(seq_vect_inf, True).to(self.device)
            user_vec_np = user_vec.cpu().numpy().reshape((-1,)).tolist()

        response = self.qdrant_client.query_points(
            collection_name="dssm_items",
            query=user_vec_np,
            limit=500
        )

        candidate_ids = [hit.payload["asin"] for hit in response.points]
        user_scores = [hit.score for hit in response.points]
        candidate_ids_str = [str(c_id) for c_id in candidate_ids if c_id is not None]
        real_history_ids_str = [str(h_id) for h_id in real_history_ids]

        all_required_items = list(set(candidate_ids_str + real_history_ids_str))
        all_entities = [{"candidate_item_id": i_id} for i_id in all_required_items]
        
        items_features_res = self.feast_store.get_online_features(
            features=[
                "item_features:cand_main_cat",
                "item_features:cand_sub_cat",
                "item_features:cand_amazon_rating",
                "item_features:cand_amazon_rat_num",
                "item_features:cand_title"
            ],
            entity_rows=all_entities
        ).to_dict()
        
        items_meta_map = {}
        for i, item_id in enumerate(items_features_res["candidate_item_id"]):
            items_meta_map[item_id] = {
                "main_cat": items_features_res["cand_main_cat"][i] or "PAD",
                "sub_cat": items_features_res["cand_sub_cat"][i] or "PAD",
                "rating": items_features_res["cand_amazon_rating"][i] or 4.0,
                "rat_num": items_features_res["cand_amazon_rat_num"][i] or 0,
                "title": items_features_res["cand_title"][i] or ""
            }

        hist_sub_cats = [items_meta_map.get(idx, {}).get("sub_cat", "PAD") for idx in real_history_ids]
        hist_ratings = [
            items_meta_map.get(idx, {}).get("rating", 4.0) 
            for idx in real_history_ids if idx in items_meta_map
        ]
        
        hist_mean_rating = np.mean(hist_ratings) if hist_ratings else 4.0
        real_hist_len = max(len(real_history_ids), 1)
        
        hist_titles_combined = " ".join([items_meta_map.get(idx, {}).get("title", "") for idx in real_history_ids]).lower()
        hist_words = set(hist_titles_combined.split())

        boosting_matrix = []
        for i, cand_id in enumerate(candidate_ids):
            cand_meta = items_meta_map.get(cand_id, {"main_cat": "PAD", "sub_cat": "PAD", "rating": 4.0, "rat_num": 0, "title": ""})
            
            rank = i + 1
            if rank <= 100:
                dssm_zone = "hot"
            elif rank <= 200:
                dssm_zone = "warm"
            else:
                dssm_zone = "cold_candidate"

            cat_share = float(hist_sub_cats.count(cand_meta["sub_cat"])) / real_hist_len
            rating_diff = float(cand_meta["rating"] - hist_mean_rating)
            
            cand_words = set(cand_meta["title"].lower().split())
            if not cand_words or not hist_words:
                jaccard = 0.0
            else:
                jaccard = float(len(cand_words.intersection(hist_words))) / len(cand_words.union(hist_words))

            row = [
                "0",
                user_scores[i],
                rank,
                cand_meta["main_cat"],
                cand_meta["sub_cat"],
                cand_meta["rating"],
                cand_meta["rat_num"],
                cand_meta["title"],
                dssm_zone,
                cat_share,
                rating_diff,
                jaccard
            ]
            boosting_matrix.append(row)

        b_pool = Pool(
            data=boosting_matrix,
            cat_features=[3, 4, 8],
            text_features=[7]
        )
        
        b_pred = self.boosting.predict(b_pool)
        final_ranked_items = [candidate_ids[idx] for idx in np.argsort(b_pred)[::-1]]
        
        return final_ranked_items[:n_recs]