import torch
import pandas as pd
import numpy as np
import pickle
from src.model.utils import ItemFeatureProjector, UserTower, ItemTower
from tqdm import tqdm
import re


def sample_for_boosting(df, total_users_to_sample=15000, random_state=42, cold_prop=0.20, target_prop=0.50, active_prop=0.30):
    user_counts = df["user_id"].value_counts().reset_index()
    user_counts.columns = ["user_id", "interaction_count"]

    def get_strata(count):
        if 2<=count<=10:
            return "cold"
        elif 11<=count<=39:
            return "target"
        else:
            return "active"
    
    user_counts["strata"] = user_counts["interaction_count"].apply(get_strata)

    proportions = {
        'cold': int(total_users_to_sample * cold_prop),
        'target': int(total_users_to_sample * target_prop),
        'active': int(total_users_to_sample * active_prop)
    }

    sampled_user_ids = []
    for strata_name, required_count in proportions.items():
        strata_users = user_counts[user_counts['strata'] == strata_name]
        available_count = len(strata_users)
        
        take_count = min(required_count, available_count)
        
        if take_count > 0:
            sampled_strata = strata_users.sample(n=take_count, random_state=random_state)
            sampled_user_ids.extend(sampled_strata['user_id'].tolist())
            print(f"Из страты '{strata_name}' успешно взято: {take_count} пользователей (требовалось {required_count})")
            
    sampled_users_set = set(sampled_user_ids)
    final_df = df[df['user_id'].isin(sampled_users_set)].copy()
    
    final_df = final_df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    
    print(f"\nИтоговый датасет собран. Уникальных пользователей: {final_df['user_id'].nunique()}")
    print(f"Общее количество строк взаимодействий для этих пользователей: {final_df.shape[0]}")
    
    return final_df

def prepare_dssm_inputs(df_boosting_users, pad_token_id=0):
    user_histories = []
    user_ids = []
    
    for user_id, group in df_boosting_users.groupby('user_id'):
        items = group['parent_asin'].tolist()
        n_items = len(items)
        
        target_idx = np.random.randint(1, n_items)
        
        history_items = items[:target_idx]
        
        if len(history_items) > 10:
            history_items = history_items[-10:]
        else:
            padding_len = 10 - len(history_items)
            history_items = [pad_token_id] * padding_len + history_items
            
        user_ids.append(user_id)
        user_histories.append(history_items)
        
    dssm_prepare_df = pd.DataFrame({
        'user_id': user_ids,
        'history': user_histories,
    })
    return dssm_prepare_df

class BoostingUserDataset(torch.utils.data.Dataset):
    def __init__(self, dssm_df, item_embs):
        self.user_ids = dssm_df["user_id"].values
        self.histories = dssm_df["history"].tolist()
        self.item_embs = item_embs
    
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        hist_ids = self.histories[idx]

        user_seq = torch.from_numpy(np.array([self.item_embs[i] if i !=0 else np.zeros((2615,), dtype=np.float32) for i in hist_ids]))

        return {
            "user_id": user_id,
            "hist_vectors": user_seq
        }

def get_boosting_raw(boosting_loader, item_embs_torch, inter_dict_b, its, user_encoder, device):
    boosting_rows = []
    for batch in boosting_loader:
        user_seq = batch["hist_vectors"].to(device)
        batch_user_ids = batch['user_id']
        with torch.no_grad(): user_emb = user_encoder(user_seq)
        scores = torch.matmul(user_emb, item_embs_torch.T)
        distances, indices = torch.topk(scores, k=500, dim=-1)

        for idx in range(len(batch_user_ids)):
            u_id = batch_user_ids[idx]
            gt = set(inter_dict_b[u_id])
        
            user_distances = distances[idx].detach().cpu().numpy()
            items_rec = indices[idx].detach().cpu().numpy().reshape((-1,))
            user_candidates = set([its[i] for i in items_rec])
        
            for rank, (cand_id, score) in enumerate(zip(user_candidates, user_distances), start=1):
                target = 1 if cand_id in gt else 0
            
                boosting_rows.append({
                    'query_id': u_id,
                    'candidate_item_id': cand_id,
                    'dssm_score': score,
                    'dssm_rank': rank,
                    'target': target
                })
    df_boosting_raw = pd.DataFrame(boosting_rows)
    
    return df_boosting_raw

def parse_amazon_categories(cat_data):
    if isinstance(cat_data, str):
        cats = cat_data.split(", ")
    else:
        return "UNKNOWN_CAT", "UNKNOWN_SUB_CAT"

    main_cat = cats[1] if len(cats) > 1 else "UNKNOWN_CAT"
    sub_cat = cats[-1] if len(cats) > 2 else main_cat
    return main_cat, sub_cat

def build_final_boosting_dataset(df_boosting_raw, df_items_meta, dssm_df):
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
    for user_id in tqdm(unique_users):
        hist_items = dssm_df[dssm_df["user_id"] == user_id]["history"].tolist()[0]

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

def main():
    interaction_dataset = pd.read_csv("data/processed/interactions_train.csv")
    interaction_dataset = interaction_dataset[["user_id", "parent_asin", "rating", "timestamp"]]
    with open("data/processed/items_full_CLIP_1.pickle", "rb") as f: item_embeddings = pickle.load(f)
    interaction_dataset = interaction_dataset[interaction_dataset["parent_asin"].isin(list(item_embeddings.keys()))]

    item_per_user = interaction_dataset.groupby("user_id")["rating"].size()
    interaction_dataset_train = interaction_dataset[interaction_dataset["user_id"].isin(item_per_user[item_per_user >= 50].index.values)].reset_index(drop=True)
    train_items = interaction_dataset_train["parent_asin"].unique()

    interaction_dataset_b = interaction_dataset[interaction_dataset["parent_asin"].isin(train_items)].reset_index(drop=True)
    item_per_user_b = interaction_dataset_b.groupby("user_id")["rating"].size()
    interaction_dataset_b = interaction_dataset_b[interaction_dataset_b["user_id"].isin(item_per_user_b[item_per_user_b >= 2].index.values)].reset_index(drop=True)

    sampled_df = sample_for_boosting(interaction_dataset_b)
    sampled_df_test = sample_for_boosting(interaction_dataset_b, 5000, 114, 0.5, 0.5, 0)

    dssm_df = prepare_dssm_inputs(sampled_df)
    dssm_df_test = prepare_dssm_inputs(sampled_df_test)

    boosting_user_dataset = BoostingUserDataset(dssm_df, item_embeddings)
    boosting_loader = torch.utils.data.DataLoader(boosting_user_dataset, batch_size=256, shuffle=False)
    boosting_user_dataset_test = BoostingUserDataset(dssm_df_test, item_embeddings)
    boosting_loader_test = torch.utils.data.DataLoader(boosting_user_dataset_test, batch_size=256, shuffle=False)

    device = torch.device("cuda")
    image_text_dim = 512
    als_dim = 100
    cat_dim = 2615 - 2*image_text_dim - als_dim
    proj_m = ItemFeatureProjector(als_dim, image_text_dim, image_text_dim, cat_dim)
    user_encoder = UserTower(proj_m)
    item_encoder = ItemTower(proj_m)

    item_weights = torch.load("models/fps_loss_transformer_item_test_1.pth")
    user_weights = torch.load("models/fps_loss_transformer_user_test_1.pth")

    item_encoder.load_state_dict(item_weights)
    item_encoder.to(device)
    item_encoder.eval()
    user_encoder.load_state_dict(user_weights)
    user_encoder.to(device)
    user_encoder.eval()

    with torch.no_grad(): items_embs = item_encoder(torch.from_numpy(np.asarray(list(item_embeddings.values()))).to(device))

    item_embs_torch = torch.tensor(items_embs).to(device)
    its = list(item_embeddings.keys())
    inter_dict_b = interaction_dataset_b.sort_values("timestamp")[["user_id", "parent_asin"]].groupby("user_id")["parent_asin"].apply(list).to_dict()
    df = get_boosting_raw(boosting_loader, item_embs_torch, inter_dict_b, its, user_encoder, device)
    df_test = get_boosting_raw(boosting_loader_test, item_embs_torch, inter_dict_b, its, user_encoder, device)
    print(df["target"].sum())

    item_dataset = pd.read_csv("data/interim/items.csv")

    df_f = build_final_boosting_dataset(df, item_dataset, dssm_df)
    df_f_test = build_final_boosting_dataset(df_test, item_dataset, dssm_df_test)
    df_f.to_csv("data/processed/boosting_dataset.csv")
    df_f_test.to_csv("data/processed/boosting_dataset_test.csv")

if __name__ == "__main__":
    main()