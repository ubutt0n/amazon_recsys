import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


class UserEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, embedding_dim):
        super(UserEncoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = h_dim
        layers.append(torch.nn.Linear(prev_dim, embedding_dim))
        self.network = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        emb = self.network(x)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb
    

class ItemEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, embedding_dim):
        super(ItemEncoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = h_dim
        layers.append(torch.nn.Linear(prev_dim, embedding_dim))
        self.network = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        emb = self.network(x)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb


class FpsLossDataset(torch.utils.data.Dataset):
    def __init__(self, interaction_dataset, user_embeddings, item_embeddings):
        self.interactions = interaction_dataset[["user_id", "parent_asin"]].values
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_id, item_id = self.interactions[idx]

        return self.user_embeddings[user_id], self.item_embeddings[item_id]


def generate_triplets(interaction_dataset):
    all_items = set(interaction_dataset["parent_asin"].unique())
    grouped = interaction_dataset.groupby("user_id")
    triplets = []
    for user, group in tqdm(grouped):
        bought_items = group["parent_asin"].values
        for _ in range(min(10, len(bought_items))):
            positive = random.choice(bought_items.tolist())
            negative = random.choice(list(all_items - set(bought_items)))

            triplets.append((user, positive, negative))

    return triplets


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, triplets, user_embeddings, item_embeddings):
        self.triplets = triplets
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_id, positive_id, negative_id = self.triplets[idx]

        return self.user_embeddings[anchor_id], self.item_embeddings[positive_id], self.item_embeddings[negative_id]


def cosine_distance(x1, x2):
    return 1 - torch.nn.functional.cosine_similarity(x1, x2, dim=1)