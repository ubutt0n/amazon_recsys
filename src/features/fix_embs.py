import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


'''with open("data/processed/users_als_embeddings_1.pickle", "rb") as f: als_item_embs = pickle.load(f)
with open("data/processed/users_full_CLIP.pickle", "rb") as f1: item_f_embs = pickle.load(f1)


new_item_embs = {}
items = list(item_f_embs.keys())

for i in tqdm(items):
    new_item_embs[i] = np.concatenate((item_f_embs[i][0:2515], als_item_embs[i]), dtype=np.float32)

with open("data/processed/users_full_CLIP_1.pickle", "wb") as f: pickle.dump(new_item_embs, f)'''

with open("data/processed/items_full_CLIP_1.pickle", "rb") as f1: item_f_embs = pickle.load(f1)

new_item_embs = {}
items = list(item_f_embs.keys())

for i in tqdm(items):
    new_item_embs[i] = item_f_embs[i][0:2515]

with open("data/processed/items_full_CLIP_wo_als.pickle", "wb") as f: pickle.dump(new_item_embs, f)