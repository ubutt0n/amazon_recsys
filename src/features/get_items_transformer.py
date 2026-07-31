from src.model.utils import ItemFeatureProjector, ItemTower, UserTower
import torch
import pandas as pd
import pickle
import numpy as np


def main():
    with open("data/processed/items_full_CLIP_1.pickle", "rb") as f: item_embeddings = pickle.load(f)

    image_text_dim = 512
    als_dim = 100
    cat_dim = 2615 - 2*image_text_dim - als_dim
    proj_m = ItemFeatureProjector(als_dim, image_text_dim, image_text_dim, cat_dim)
    item_encoder = ItemTower(proj_m)

    item_weights = torch.load("models/fps_loss_transformer_item_test_1.pth")
    item_encoder.load_state_dict(item_weights)
    item_encoder.eval()

    with torch.no_grad(): items_embs_dssm = item_encoder(torch.from_numpy(np.asarray(list(item_embeddings.values())))).numpy()
    with torch.no_grad(): items_embs_transformer_input = item_encoder.item_projection(torch.from_numpy(np.asarray(list(item_embeddings.values())))).numpy()

    dssm_dict = {}
    t_input_dict = {}

    for i, k in enumerate(item_embeddings.keys()):
        dssm_dict[k] = items_embs_dssm[i]

    for i, k in enumerate(item_embeddings.keys()):
        t_input_dict[k] = items_embs_transformer_input[i]

    with open("data/processed/items_dssm.pickle", "wb") as f: pickle.dump(dssm_dict, f)
    with open("data/processed/items_t_input.pickle", "wb") as f: pickle.dump(t_input_dict, f)

if __name__ == "__main__":
    main()