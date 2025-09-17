import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset
from rectools.metrics import MAP
from rectools.metrics.ranking import NDCG
import numpy as np
import torch
from tqdm import tqdm


def get_recs(
        users: np.ndarray,
        user_encoder: torch.nn.Module,
        user_embeddings: dict,
        device: torch.device,
        items_embs: np.ndarray,
        its: list[str]
) -> pd.DataFrame:

    user_m = []
    with torch.no_grad():
        for user in users:
            user_m.append(user_encoder(torch.from_numpy(user_embeddings[user]).reshape(1, 2615).to(device)).squeeze(-1).detach().cpu().numpy())
    user_m = np.asarray(user_m).squeeze()
    scores = (user_m @ items_embs.T)
    items_rec = scores.argsort()[:, ::-1][:, :100]
    rows = []
    for user, recs in zip(users, items_rec):
        for rank, item in enumerate(recs, start=1):
            rows.append({Columns.User: user, Columns.Item: its[item], Columns.Rank: rank})
    r = pd.DataFrame(rows, columns=[Columns.User, Columns.Item, Columns.Rank])
    return r

def get_recs_by_batch(
        users: np.ndarray,
        split_size: int,
        user_encoder: torch.nn.Module,
        user_embeddings: dict,
        device: torch.device,
        items_embs: np.ndarray,
        its: list[str]
) -> pd.DataFrame:

    f_batch = get_recs(users[0:split_size], user_encoder, user_embeddings, device, items_embs, its)
    for i in tqdm(range(split_size, len(users), split_size)):
        batch = get_recs(users[i::], user_encoder, user_embeddings, device, items_embs, its) if i + split_size >= len(users) else get_recs(users[i:i+split_size], user_encoder, user_embeddings, device, items_embs, its)
        f_batch = pd.concat([f_batch, batch], ignore_index=True)
    return f_batch

def calculate_map(
        n: int,
        recs: pd.DataFrame,
        inter_data: pd.DataFrame
) -> float:
    
    m = MAP(n)
    return m.calc(recs, inter_data)

def calculate_ndcg(
        n: int,
        recs: pd.DataFrame,
        inter_data: pd.DataFrame
) -> float:
    
    nd = NDCG(n)
    return nd.calc(recs, inter_data)