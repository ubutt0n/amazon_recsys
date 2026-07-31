import mlflow
import numpy as np
import pickle
import torch
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import typing as tp
from minio import Minio
from sqlalchemy import create_engine
import re
import pandas as pd
from src.model.utils import ItemFeatureProjector, UserTower
from catboost import Pool, CatBoostRanker


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

        
def parse_amazon_categories(cat_data):
    if isinstance(cat_data, str):
        cats = cat_data.split(", ")
    else:
        return "UNKNOWN_CAT", "UNKNOWN_SUB_CAT"

    main_cat = cats[1] if len(cats) > 1 else "UNKNOWN_CAT"
    sub_cat = cats[-1] if len(cats) > 2 else main_cat
    return main_cat, sub_cat

def build_final_boosting_dataset(df_boosting_raw, df_items_meta, history):
    meta = df_items_meta.copy()

    parsed_cats = meta["categories"].apply(parse_amazon_categories)
    meta["item_main_cat"] = [c[0] for c in parsed_cats]
    meta["item_sub_cat"] = [c[1] for c in parsed_cats]

    item_key = "parent_asin"
    item_to_main_cat = meta.set_index(item_key)["item_main_cat"].to_dict()
    item_to_sub_cat = meta.set_index(item_key)["item_sub_cat"].to_dict()
    item_to_rating = meta.set_index(item_key)["average_rating"].to_dict()
    item_to_rat_num = meta.set_index(item_key)["rating_number"].to_dict()
    item_to_title = meta.set_index(item_key)["title"].to_dict()

    unique_users = df_boosting_raw["query_id"].unique()

    user_history_features = {}
    for user_id in unique_users:
        hist_items = history
        
        hist_sub_cats = [item_to_sub_cat.get(idx, "PAD") for idx in hist_items]
        hist_ratings = [
            item_to_rating.get(idx, np.nan)
            for idx in hist_items
            if not pd.isna(item_to_rating.get(idx, np.nan))
        ]

        hist_titles_combined = " ".join(
            [str(item_to_title.get(idx, "")) for idx in hist_items]
        ).lower()

        user_history_features[user_id] = {
            "hist_sub_cats_dict": {
                c: hist_sub_cats.count(c) for c in set(hist_sub_cats)
            },
            "hist_mean_rating": np.mean(hist_ratings) if hist_ratings else 4.0,
            "hist_titles_words": set(
                [w for w in re.findall(r"\w+", hist_titles_combined) if len(w) > 2]
            ),
            "real_hist_len": max(len(hist_items), 1),
        }
    
    df = df_boosting_raw.copy()

    df["cand_main_cat"] = (
        df["candidate_item_id"].map(item_to_main_cat).fillna("PAD")
    )
    df["cand_sub_cat"] = (
        df["candidate_item_id"].map(item_to_sub_cat).fillna("PAD")
    )
    df["cand_amazon_rating"] = (
        df["candidate_item_id"].map(item_to_rating).fillna(4.0)
    )
    df["cand_amazon_rat_num"] = (
        df["candidate_item_id"].map(item_to_rat_num).fillna(0)
    )
    df["cand_title"] = df["candidate_item_id"].map(item_to_title).fillna("")

    df["dssm_zone"] = pd.cut(
        df["dssm_rank"],
        bins=[0, 100, 200, 501],
        labels=["hot", "warm", "cold_candidate"],
    )


    df["user_profile"] = df["query_id"].map(user_history_features)

    empty_profile = {
        "hist_sub_cats_dict": {},
        "hist_mean_rating": 4.0,
        "hist_titles_words": set(),
        "real_hist_len": 1,
    }
    df["user_profile"] = df["user_profile"].apply(
        lambda x: x if isinstance(x, dict) else empty_profile
    )

    df["cat_share_in_history"] = df.apply(
        lambda r: (
            r["user_profile"]["hist_sub_cats_dict"].get(r["cand_sub_cat"], 0)
            / r["user_profile"]["real_hist_len"]
        ),
        axis=1,
    ).astype(np.float32)

    df["rating_diff_from_hist"] = (
        df["cand_amazon_rating"]
        - df["user_profile"].apply(lambda x: x["hist_mean_rating"])
    ).astype(np.float32)


    def calculate_jaccard_v2(row):
        cand_words = set(
            [
                w
                for w in re.findall(r"\w+", str(row["cand_title"]).lower())
                if len(w) > 2
            ]
        )
        hist_words = row["user_profile"]["hist_titles_words"]
        if not cand_words or not hist_words:
            return 0.0
        intersection = len(cand_words.intersection(hist_words))
        union = len(cand_words.union(hist_words))
        return float(intersection) / union

    df["title_jaccard_similarity"] = df.apply(
        calculate_jaccard_v2, axis=1
    ).astype(np.float32)

    df_final = df.drop(columns=["user_profile"])
    return df_final

class UserDataT(BaseModel):
    items: tp.List[str]

def load_transformer_model():
    device = torch.device("cuda")
    run_id = "31140806f171423792c3e72730a93578"
    artifact_path = "model_weights/fps_loss_transformer_user_test_1.pth"
    local_weight_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, 
        artifact_path=artifact_path,
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
    artifact_path = "model_weights/boosting_boosting2.cbm"
    local_weight_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, 
        artifact_path=artifact_path,
        dst_path="/tmp/models"
    )
    boosting = CatBoostRanker()
    boosting.load_model(local_weight_path)
    return boosting


class ModelTransformer():
    def __init__(self):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        self.device = torch.device("cuda")
        self.user_encoder = load_transformer_model()
        self.boosting = load_catboost_model()
        self.items_meta = pd.read_csv("items.csv")
        with open("items_t_input.pickle", "rb") as f: self.items_t_inp = pickle.load(f)
        with open("items_dssm.pickle", "rb") as f: items_dssm = pickle.load(f)
        self.items_dssm_t = torch.from_numpy(np.asarray(list(items_dssm.values()))).to(self.device)
        self.its = list(items_dssm.keys())

    def recommend(self, user_data: list, n_recs: int):
        user_history = user_data.items
        seq_vect_inf = []
        for item in user_history:
            if item != "0":
                seq_vect_inf.append(self.items_t_inp[item])
            else:
                seq_vect_inf.append(np.zeros((128,), dtype=np.float32))
        seq_vect_inf = torch.from_numpy(np.asarray(seq_vect_inf).reshape(1, 10, -1)).to(self.device)
        with torch.no_grad(): user_vec = self.user_encoder(seq_vect_inf, True).to(self.device)
        distances, indices = torch.topk(torch.matmul(user_vec, self.items_dssm_t.T), k=500, dim=-1)
        items_rec = indices.detach().cpu().numpy().reshape((-1,))

        user_candidates = set([self.its[i] for i in items_rec])
        user_distances = distances.detach().cpu().numpy().reshape((-1,))
        boosting_rows = []
        for rank, (cand_id, score) in enumerate(zip(user_candidates, user_distances), start=1):
            boosting_rows.append(
                {
                    'candidate_item_id': cand_id,
                    'dssm_score': score,
                    'dssm_rank': rank,
                }
            )
        boosting_df_raw = pd.DataFrame(boosting_rows)
        boosting_df_raw["query_id"] = "0"     

        boosting_inp = build_final_boosting_dataset(boosting_df_raw, self.items_meta, user_history)

        text_features = ["cand_title"]
        cat_features = ["cand_main_cat", "cand_sub_cat", "dssm_zone"]

        b_pool = Pool(
            data=boosting_inp.drop(columns=["query_id", "candidate_item_id"]),
            group_id=boosting_inp["query_id"],
            cat_features=cat_features,
            text_features=text_features,
        )

        b_pred = self.boosting.predict(b_pool)

        res = np.asarray(boosting_inp["candidate_item_id"].to_list())[np.argsort(b_pred)][::-1][0:n_recs]
        return res.tolist()