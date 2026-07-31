import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

interaction_dataset = pd.read_csv("data/processed/interactions_train.csv")
items_dataset = pd.read_csv("data/interim/items.csv")

with open("data/processed/items_base.pickle", "rb") as f: item_image_embeddings = pickle.load(f)
with open("data/processed/users_als_embeddings.pickle", "rb") as f: als_user_embs = pickle.load(f)
with open("data/processed/items_als_embeddings.pickle", "rb") as f: als_item_embs = pickle.load(f)

items_dataset["categories"] = list(map(lambda x: x.split(", "), items_dataset["categories"].values))
mlb = MultiLabelBinarizer(sparse_output=True)
items_dataset = items_dataset.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(items_dataset.pop('categories')),
                index=items_dataset.index,
                columns=mlb.classes_))
items_dataset = items_dataset.drop(columns=["title", "average_rating", "rating_number", "images"])
encoded_cats = items_dataset.drop(columns=["parent_asin"]).values
encoded_cats_dict = {key: encoded_cats[i] for i, key in enumerate(items_dataset["parent_asin"].values)}

'''def get_user_emb(user):
    it_user = interaction_dataset[interaction_dataset["user_id"] == user]
    user_items = it_user["parent_asin"].values
    user_ratings = it_user["rating"].values
    user_image_emb = 0
    user_cat_emb = 0
    for i, item in enumerate(user_items):
        user_image_emb += item_image_embeddings[item] * user_ratings[i]
        user_cat_emb += encoded_cats_dict[item] * user_ratings[i]
    user_image_emb = user_image_emb / np.sum(user_ratings)
    user_cat_emb = user_cat_emb / np.sum(user_ratings)
    user_als_emb = als_user_embs[user]
    
    user_emb = np.concatenate((user_image_emb, user_cat_emb, user_als_emb), dtype=np.float32)
    return user_emb

def get_item_emb(item):
    return np.concatenate((item_image_embeddings[item], encoded_cats_dict[item], als_item_embs[item]), dtype=np.float32)'''

user_f_embs = {}

#items = interaction_dataset["parent_asin"].unique()
#for item in tqdm(items):
#    item_f_embs[item] = get_item_emb(item)

user_grouped = interaction_dataset[["user_id", "parent_asin", "rating"]].groupby("user_id")

for user, inter in tqdm(user_grouped):
    user_items = inter["parent_asin"].values
    user_ratings = inter["rating"].values
    user_image_emb = 0
    user_cat_emb = 0
    for i, item in enumerate(user_items):
        user_image_emb += item_image_embeddings[item] * user_ratings[i]
        user_cat_emb += encoded_cats_dict[item] * user_ratings[i]
    user_image_emb = user_image_emb / np.sum(user_ratings)
    user_cat_emb = user_cat_emb / np.sum(user_ratings)
    user_als_emb = als_user_embs[user]
    
    user_emb = np.concatenate((user_image_emb, user_cat_emb, user_als_emb), dtype=np.float32)
    user_f_embs[user] = user_emb

print(len(user_f_embs))
print(interaction_dataset["user_id"].unique().shape)

#for user in tqdm(interaction_dataset["user_id"].unique()):
#    user_f_embs[user] = get_user_emb(user)

with open("data/processed/users_full.pickle", "wb") as f: pickle.dump(user_f_embs, f)