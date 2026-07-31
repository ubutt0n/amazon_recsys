import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from tqdm import tqdm
import random
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict
import math

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#print(device)

interaction_dataset = pd.read_csv("data/processed/interactions_train.csv")
#items_dataset = pd.read_csv("data/interim/items.csv")

ratings_per_user = interaction_dataset.groupby('user_id')['rating'].count().sort_values(ascending=False)
remove_users = []

for user_id, num_ratings in ratings_per_user.items():
    if num_ratings < 10:
        remove_users.append(user_id)

interaction_dataset = interaction_dataset.loc[ ~ interaction_dataset['user_id'].isin(remove_users)]
#interaction_dataset = interaction_dataset[interaction_dataset["rating"] >= 3]


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

user_input_dim = 3639
item_input_dim = 3639
hidden_dims = [300, 300]
embedding_dim = 128

user_encoder = UserEncoder(user_input_dim, hidden_dims, embedding_dim)
#user_encoder.load_state_dict(torch.load("user_emb.pth", weights_only=True))
user_encoder.to(device)
item_encoder = ItemEncoder(item_input_dim, hidden_dims, embedding_dim)
#item_encoder.load_state_dict(torch.load("item_emb.pth", weights_only=True))
item_encoder.to(device)

def clip_full_product_softmax_loss(similarity_logits):
    """
    similarity_logits: Tensor of shape (batch_size_image, batch_size_text)
                       Contains similarity scores (e.g., cosine similarity)
                       between image and text embeddings.

    Returns:
        Scalar loss combining image-to-text and text-to-image cross entropy losses.
    """
    batch_size = similarity_logits.size(0)
    assert similarity_logits.size(0) == similarity_logits.size(1), "Batch sizes for image and text must be equal"

    targets = torch.arange(batch_size).to(similarity_logits.device)  # Correct pairs on diagonal

    # Image-to-text loss (row-wise softmax + cross-entropy)
    loss_i2t = torch.nn.functional.cross_entropy(similarity_logits, targets)

    # Text-to-image loss (column-wise softmax + cross-entropy)
    loss_t2i = torch.nn.functional.cross_entropy(similarity_logits.t(), targets)

    # Combine losses symmetrically
    loss = (loss_i2t + loss_t2i) / 2
    return loss

'''with open("data/processed/items_base.pickle", "rb") as f: item_image_embeddings = pickle.load(f)
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

def get_user_emb(user):
    user_items = interaction_dataset[interaction_dataset["user_id"] == user]["parent_asin"].values
    user_ratings = interaction_dataset[interaction_dataset["user_id"] == user]["rating"].values
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

with open("data/processed/items_full.pickle", "rb") as f: item_embeddings = pickle.load(f)
with open("data/processed/users_full.pickle", "rb") as f: user_embeddings = pickle.load(f)

def get_user_emb(user):
    return user_embeddings[user]

def get_item_emb(item):
    return item_embeddings[item]

class ClipInteractionDataset(torch.utils.data.Dataset):
    def __init__(self, max_items_per_user=None):
        """
        interactions: list<Tuple(user_id, item_id, rating)> положительных взаимодействий
        user_features: dict user_id -> feature vector (tensor или np.array)
        item_features: dict item_id -> feature vector (tensor или np.array)
        max_items_per_user: максимальное количество взаимодействий пользователя в датасете (None - все)
        """
        self.max_items_per_user = max_items_per_user

        # Группируем по пользователям
        self.user_to_items = defaultdict(list)
        for user_id, item_id in interaction_dataset[["user_id", "parent_asin"]].values:
            self.user_to_items[user_id].append(item_id)

        self.user_items = []
        for user, items in self.user_to_items.items():
            if max_items_per_user is not None and len(items) > max_items_per_user:
                items = random.sample(items, max_items_per_user)
            for item in items:
                self.user_items.append((user, item))

        # Создаем словарь для быстрого поиска user_id по индексу для Sampler
        self.idx_to_user = [ui[0] for ui in self.user_items]

    def __len__(self):
        return len(self.user_items)

    def __getitem__(self, idx):
        user_id, item_id = self.user_items[idx]

        return get_user_emb(user_id), get_item_emb(item_id), user_id


class UniqueUserBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        """
        Формируем батчи без повторения пользователей внутри.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.idx_to_user = dataset.idx_to_user

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        
        batch = []
        users_in_batch = set()
        for idx in self.indices:
            user = self.idx_to_user[idx]
            if user not in users_in_batch:
                batch.append(idx)
                users_in_batch.add(user)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    users_in_batch = set()
        if batch:
            yield batch

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


class InteractionsDataset(torch.utils.data.Dataset):
    def __init__(self):
        """
        triplets: список кортежей (user, anchor_item, positive_item, negative_item)
        item_to_image_path: dict или функция, возвращающая путь к изображению товара по item_id
        transform: torchvision.transforms для предобработки изображений
        """
        self.data = interaction_dataset
        self.users = interaction_dataset["user_id"].unique()
        self.user_to_items = defaultdict(list)
        for user_id, item_id in interaction_dataset[["user_id", "parent_asin"]].values:
            self.user_to_items[user_id].append(item_id)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        pos_items = self.user_to_items[user_id]
        
        # Случайно выбираем один положительный item для пользователя
        pos_item_id = random.choice(pos_items)

        return get_user_emb(user_id), get_item_emb(pos_item_id)


class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        """
        Бэтч-сэмплер, который гарантирует, что в одном батче не будет повторяющихся пользователей.
        dataset - экземпляр UserPositiveDataset, у которого __len__ = число уникальных пользователей
        shuffle - перемешивать пользователей перед формированием батчей
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.user_indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.user_indices)
        # Разбиваем пользователей на батчи
        batches = [self.user_indices[i:i + self.batch_size] for i in range(0, len(self.user_indices), self.batch_size)]
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


dataset = ClipInteractionDataset()
batch_size = 32
sampler = UniqueUserBatchSampler(dataset, batch_size=batch_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)

optimizer = torch.optim.Adam(list(user_encoder.parameters()) + list(item_encoder.parameters()), lr=1e-3)


def train():
    user_encoder.train()
    item_encoder.train()
    for epoch in range(10):
        total_loss = 0
        for i, embs in enumerate(tqdm(dataloader)):
            batch_user_feats, batch_item_feats, _ = embs
            batch_user_feats = batch_user_feats.to(device)
            batch_item_feats = batch_item_feats.to(device)

            optimizer.zero_grad()

            user_emb = user_encoder(batch_user_feats)
            item_emb = item_encoder(batch_item_feats)

            similarity_matrix = user_emb @ item_emb.t()

            target = torch.arange(similarity_matrix.size(0), device=device)

            loss_user_to_item = torch.nn.functional.cross_entropy(similarity_matrix, target)
            loss_item_to_user = torch.nn.functional.cross_entropy(similarity_matrix.t(), target)

            loss = (loss_user_to_item + loss_item_to_user) / 2

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{10}], Avg Loss: {avg_loss:.4f}")
        #torch.save(user_encoder.state_dict(), "user_emb_fps_loss_full2.pth")
        #torch.save(item_encoder.state_dict(), "item_emb_fps_loss_full2.pth")


if __name__ == "__main__":
    train()
    torch.save(user_encoder.state_dict(), "models/user_emb_fps_loss_full4.pth")
    torch.save(item_encoder.state_dict(), "models/item_emb_fps_loss_full4.pth")