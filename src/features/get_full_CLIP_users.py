import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


with open("data/processed/items_full_CLIP.pickle", "rb") as f: item_f_embs = pickle.load(f)
with open("data/processed/users_als_embeddings.pickle", "rb") as f: als_user_embs = pickle.load(f)

interactions_dataset = pd.read_csv("data/processed/interactions_train.csv")

user_f_embs = {}

interactions_dataset = interactions_dataset[interactions_dataset["parent_asin"].isin(list(item_f_embs.keys()))]
user_grouped = interactions_dataset[["user_id", "parent_asin", "rating"]].groupby("user_id")

for user, inter in tqdm(user_grouped):
    user_items = inter["parent_asin"].values
    user_ratings = inter["rating"].values

    user_item_emb = 0
    for i, item in enumerate(user_items):
        user_item_emb += user_ratings[i] * item_f_embs[item][0:2515]
    user_item_emb = user_item_emb / np.sum(user_ratings)
    user_als_emb = als_user_embs[user]

    user_f_embs[user] = np.concatenate((user_item_emb, user_als_emb), dtype=np.float32)

print(len(user_f_embs))
print(interactions_dataset["user_id"].unique().shape)

with open("data/processed/users_full_CLIP.pickle", "wb") as f: pickle.dump(user_f_embs, f)