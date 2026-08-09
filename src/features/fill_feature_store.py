import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sqlalchemy import create_engine, types
import os
from dotenv import load_dotenv


load_dotenv()

def parse_amazon_categories(cat_data):
    if isinstance(cat_data, str):
        cats = cat_data.split(", ")
    else:
        return "UNKNOWN_CAT", "UNKNOWN_SUB_CAT"

    main_cat = cats[1] if len(cats) > 1 else "UNKNOWN_CAT"
    sub_cat = cats[-1] if len(cats) > 2 else main_cat
    return main_cat, sub_cat

df_items = pd.read_csv("data/interim/items.csv")

with open("data/processed/items_t_input.pickle", "rb") as f:
    items_t_inp = pickle.load(f)

parsed_cats = df_items["categories"].apply(parse_amazon_categories)
df_items["cand_main_cat"] = [c[0] for c in parsed_cats]
df_items["cand_sub_cat"] = [c[1] for c in parsed_cats]

df_items = df_items.rename(columns={"parent_asin": "candidate_item_id"})
df_items = df_items.rename(columns={"average_rating": "cand_amazon_rating", "rating_number": "cand_amazon_rat_num"})

df_final = df_items[["candidate_item_id", "cand_main_cat", "cand_sub_cat", "cand_amazon_rating", "cand_amazon_rat_num", "title"]].copy()
df_final = df_final.rename(columns={"title": "cand_title"})

def get_vector_as_json(item_id):
    vec = items_t_inp.get(item_id, np.zeros((128,), dtype=np.float32))
    return json.dumps(vec.tolist())

df_final["transformer_input_vector"] = df_final["candidate_item_id"].apply(get_vector_as_json)

df_final["event_timestamp"] = datetime.now(timezone.utc)
df_final["created_timestamp"] = datetime.now(timezone.utc)

df_final["cand_main_cat"] = df_final["cand_main_cat"].fillna("PAD")
df_final["cand_sub_cat"] = df_final["cand_sub_cat"].fillna("PAD")
df_final["cand_amazon_rating"] = df_final["cand_amazon_rating"].fillna(4.0).astype(np.float32)
df_final["cand_amazon_rat_num"] = df_final["cand_amazon_rat_num"].fillna(0).astype(np.int64)
df_final["cand_title"] = df_final["cand_title"].fillna("").astype(str)

connection_string = f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('DATABASE_HOST')}/{os.getenv('POSTGRES_DB')}"
engine = create_engine(connection_string)

df_final.to_sql(
    name="items_feast_table",
    con=engine,
    if_exists="replace",
    index=False,
    dtype={
        "candidate_item_id": types.VARCHAR(255),
        "cand_main_cat": types.VARCHAR(255),
        "cand_sub_cat": types.VARCHAR(255),
        "cand_amazon_rating": types.Float(precision=2),
        "cand_amazon_rat_num": types.BigInteger(),
        "cand_title": types.Text(),
        "transformer_input_vector": types.Text(),
        "event_timestamp": types.TIMESTAMP,
        "created_timestamp": types.TIMESTAMP
    }
)