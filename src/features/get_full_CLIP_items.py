import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


items_dataset = pd.read_csv("data/interim/items.csv")
interaction_dataset = pd.read_csv("data/processed/interactions_train.csv")

with open("data/processed/items_als_embeddings.pickle", "rb") as f: als_item_embs = pickle.load(f)
items_dataset["categories"] = list(map(lambda x: x.split(", "), items_dataset["categories"].values))
mlb = MultiLabelBinarizer(sparse_output=True)
items_dataset = items_dataset.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(items_dataset.pop('categories')),
                index=items_dataset.index,
                columns=mlb.classes_))
encoded_cats = items_dataset.drop(columns=["title", "average_rating", "rating_number", "images", "parent_asin"]).values
encoded_cats_dict = {key: encoded_cats[i] for i, key in enumerate(items_dataset["parent_asin"].values)}

items_dataset = items_dataset[["parent_asin", "title"]]

item_f_embs = {}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

unique_items = interaction_dataset["parent_asin"].unique()
items_dataset = items_dataset.loc[items_dataset["parent_asin"].isin(unique_items), ["parent_asin", "title"]].values

for item, title in tqdm(items_dataset):
    if type(title) is not str:
        continue
    img = Image.open("data/interim/item_images/" + item + ".png").convert("RGB")
    inputs = processor(text=[title], images=img, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds

        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)

        item_f_embs[item] = np.concatenate((img_emb.cpu().numpy().squeeze(), txt_emb.cpu().numpy().squeeze(), encoded_cats_dict[item], als_item_embs[item]), dtype=np.float32)

with open("data/processed/items_full_CLIP.pickle", "wb") as f: pickle.dump(item_f_embs, f)