import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from tqdm import tqdm
import random
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from rectools import Columns
from rectools.dataset import Dataset
from rectools.metrics import MAP
from rectools.metrics.ranking import NDCG

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#print(device)

interaction_dataset = pd.read_csv("data/processed/interactions_train.csv")
items_dataset = pd.read_csv("data/interim/items.csv")

ratings_per_user = interaction_dataset.groupby('user_id')['rating'].count().sort_values(ascending=False)
remove_users = []

for user_id, num_ratings in ratings_per_user.items():
    if num_ratings >= 50:
        remove_users.append(user_id)

interaction_dataset = interaction_dataset.loc[ ~ interaction_dataset['user_id'].isin(remove_users)]

with open("data/processed/items_full.pickle", "rb") as f: item_embeddings = pickle.load(f)
with open("data/processed/users_full.pickle", "rb") as f: user_embeddings = pickle.load(f)

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
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # L2-нормализация
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
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # L2-нормализация
        return emb

def cosine_distance(x1, x2):
    return 1 - torch.nn.functional.cosine_similarity(x1, x2, dim=1)

user_input_dim = 3639
item_input_dim = 3639
hidden_dims = [300, 300]
embedding_dim = 128

user_encoder = UserEncoder(user_input_dim, hidden_dims, embedding_dim)
user_encoder.load_state_dict(torch.load("models/user_emb_fps_loss_full_s.pth", weights_only=True))
user_encoder.to(device)
user_encoder.eval()
item_encoder = ItemEncoder(item_input_dim, hidden_dims, embedding_dim)
item_encoder.load_state_dict(torch.load("models/item_emb_fps_loss_full_s.pth", weights_only=True))
item_encoder.to(device)
item_encoder.eval()


items_embs = item_encoder(torch.from_numpy(np.asarray(list(item_embeddings.values()))).to(device)).detach().cpu().numpy()
its = list(item_embeddings.keys())

def get_recs1(users):
    user_m = []
    for user in users:
        user_m.append(user_encoder(torch.from_numpy(user_embeddings[user]).reshape(1, 3639).to(device)).squeeze(-1).detach().cpu().numpy())
    user_m = np.asarray(user_m).squeeze()
    scores = (user_m @ items_embs.T)
    items_rec = scores.argsort()[:, ::-1][:, :100]
    rows = []
    for user, recs in zip(users, items_rec):
        for rank, item in enumerate(recs, start=1):
            rows.append({Columns.User: user, Columns.Item: its[item], Columns.Rank: rank})
    r = pd.DataFrame(rows, columns=[Columns.User, Columns.Item, Columns.Rank])
    return r

def get_recs_by_batch(users, split_size):
    f_batch = get_recs1(users[0:split_size])
    for i in tqdm(range(split_size, len(users), split_size)):
        batch = get_recs1(users[i::]) if i + split_size >= len(users) else get_recs1(users[i:i+split_size])
        f_batch = pd.concat([f_batch, batch], ignore_index=True)
    return f_batch

f_r = get_recs_by_batch(interaction_dataset["user_id"].unique()[0:10000], 1000)

inter_data = interaction_dataset[["user_id", "parent_asin", "rating"]].rename(columns={"user_id": Columns.User, "parent_asin": Columns.Item, "rating": Columns.Weight})
inter_data = inter_data.loc[inter_data["user_id"].isin(interaction_dataset["user_id"].unique()[0:10000])]

m = MAP(10)
n = NDCG(10)

print("{:.8f}".format(m.calc(f_r, inter_data)))
print("{:.8f}".format(n.calc(f_r, inter_data)))