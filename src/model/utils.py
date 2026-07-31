import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import math


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


class ItemFeatureProjector(torch.nn.Module):
    def __init__(self, dim_als, dim_img, dim_txt, dim_cat, hidden_dim=128):
        super().__init__()
        # Запоминаем размеры для правильного слайсинга тензора
        self.dims = [dim_img, dim_txt, dim_cat, dim_als]
        
        # Индивидуальные проекции для каждой модальности
        #proj_dim = hidden_dim//4
        self.proj_als = torch.nn.Sequential(torch.nn.Linear(dim_als, hidden_dim), torch.nn.LayerNorm(hidden_dim))
        self.proj_img = torch.nn.Sequential(torch.nn.Linear(dim_img, hidden_dim), torch.nn.LayerNorm(hidden_dim))
        self.proj_txt = torch.nn.Sequential(torch.nn.Linear(dim_txt, hidden_dim), torch.nn.LayerNorm(hidden_dim))
        self.proj_cat = torch.nn.Sequential(torch.nn.Linear(dim_cat, hidden_dim), torch.nn.LayerNorm(hidden_dim))
        
        # Финальная нелинейность после объединения
        #self.dense = torch.nn.Linear(hidden_dim*4, hidden_dim)
        self.activation = torch.nn.GELU()
        
    def forward(self, x):
        # x может иметь форму [Batch_Size, 2615] или [Batch_Size, Seq_Len, 2615]
        # Используем torch.split для разделения конкатенированного вектора на части
        img_x, txt_x, cat_x, als_x = torch.split(x, self.dims, dim=-1)
        
        # Проецируем каждую модальность в единое пространство hidden_dim
        out_als = self.proj_als(als_x)
        out_img = self.proj_img(img_x)
        out_txt = self.proj_txt(txt_x)
        out_cat = self.proj_cat(cat_x)
        
        # Агрегируем модальности. 
        # Сложение (Sum) предпочтительнее конкатенации, так как оно сохраняет 
        # размерность hidden_dim и работает как «голосование» разных признаков
        # .
        #combined = torch.cat([out_als, out_img, out_txt, out_cat], dim=-1)
        #out = self.dense(self.activation(combined))
        combined = out_als + out_img + out_txt + out_cat
        
        return self.activation(combined)
    

class ItemTower(torch.nn.Module):
    def __init__(self, feature_projector, hidden_dim=128, embed_dim=64):
        super().__init__()
        # Тот же самый проектор (веса разделяемые!)
        self.item_projection = feature_projector
        self.item_clip_proj = torch.nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, target_item):
        # target_item (11-й товар): [Batch_Size, 2615]
        
        # 1. Проецируем один товар: [Batch_Size, 2615] -> [Batch_Size, 128]
        x = self.item_projection(target_item)
        
        # 2. Проекция в CLIP-пространство и нормализация
        item_embeds = self.item_clip_proj(x)
        return torch.nn.functional.normalize(item_embeds, p=2, dim=-1) # [Batch_Size, 64]


class UserTower(torch.nn.Module):
    def __init__(self, feature_projector, hidden_dim=128, embed_dim=64, max_len=10, num_heads=4, num_layers=2):
        super().__init__()
        # Наш модульный проектор (из 2615 -> 128)
        self.item_projection = feature_projector
        
        self.position_embeddings = torch.nn.Embedding(max_len, hidden_dim)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim, # Теперь это строго 128
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.3,
            activation='gelu',
            batch_first=True
        )
        self.transformer_user = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.user_clip_proj = torch.nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, user_history, inference=False):
        # user_history: [Batch_Size, 10, 2615]
        
        # 1. Проецируем всю последовательность разом
        # [Batch_Size, 10, 2615] -> [Batch_Size, 10, 128]
        padding_mask = (user_history.sum(dim=-1) == 0)
        if not inference:
            x = self.item_projection(user_history)
        else:
            x = user_history
        
        # 2. Добавляем позиции
        positions = torch.arange(user_history.size(1), device=user_history.device).unsqueeze(0)
        x = x + self.position_embeddings(positions)
        
        # 3. Трансформер и пулинг по последнему токену
        user_feats = self.transformer_user(x, src_key_padding_mask=padding_mask)
        valid_mask = (~padding_mask).unsqueeze(-1).float() 
        # Зануляем выходы трансформера на позициях паддингов
        masked_feats = user_feats * valid_mask
        last_token_feat = masked_feats.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        
        #last_token_feat = user_feats[:, -1, :] # [Batch_Size, 128]
        
        # 4. Проекция в CLIP-пространство и нормализация
        user_embeds = self.user_clip_proj(last_token_feat)
        return torch.nn.functional.normalize(user_embeds, p=2, dim=-1) # [Batch_Size, 64]


class FpsLossDatasetTransformerFix(torch.utils.data.Dataset):
    def __init__(self, interactions_sampled, item_embeddings, item_id_map, interactions_dict, max_seq_len=10):
        self.interactions = interactions_sampled
        self.item_embeddings = item_embeddings
        self.item_id_map = item_id_map
        self.max_seq_len = max_seq_len
        self.interactions_dict = interactions_dict
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_id, item_id, item_idx = self.interactions[idx]
        if item_idx >= self.max_seq_len:
            user_seq_0 = self.interactions_dict[user_id][item_idx-self.max_seq_len:item_idx]
        else:
            user_seq_0 = self.interactions_dict[user_id][:item_idx]
        #user_seq = [self.item_id_map[i]+1 for i in user_seq_0]

        pad_size = self.max_seq_len - len(user_seq_0)
        padded_seq = [0] * pad_size + user_seq_0
        emb_seq = []
        for idx in padded_seq:
            if idx != 0:
                emb_seq.append(self.item_embeddings[idx])
            else:
                emb_seq.append(np.zeros((2615), dtype=np.float32))

        return item_id, np.array(emb_seq), self.item_embeddings[item_id]


class FpsLossDatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, interactions_sampled, item_embeddings, item_id_map, interactions_dict, max_seq_len=25):
        self.interactions = interactions_sampled
        self.item_embeddings = item_embeddings
        self.item_id_map = item_id_map
        self.max_seq_len = max_seq_len
        self.interactions_dict = interactions_dict
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_id, item_id, item_idx = self.interactions[idx]
        if item_idx >= self.max_seq_len:
            user_seq_0 = self.interactions_dict[user_id][item_idx-self.max_seq_len:item_idx]
        else:
            user_seq_0 = self.interactions_dict[user_id][:item_idx]
        user_seq = [self.item_id_map[i]+1 for i in user_seq_0]

        pad_size = self.max_seq_len - len(user_seq)
        padded_seq = [0] * pad_size + user_seq

        return np.array(padded_seq), self.item_embeddings[item_id]


def generate_fps_batches(interactions: dict, batch_size = 32, max_batches_per_user = 10):
    users = list(interactions.keys())
    batches = []
    for user in tqdm(users):
        for _ in range(max_batches_per_user):
            restricted_items = set()
            batch_items = set()
            batch = []

            f_item = random.choice(interactions[user].tolist())
            batch.append((user, f_item))
            batch_items.add(f_item)
            restricted_items.update(set(interactions[user]))

            while len(batch) < batch_size:
                user_1 = random.choice(users)
                bought_items = interactions[user_1]
                if not batch_items.isdisjoint(set(bought_items)): continue
                allowed = list(set(bought_items) - restricted_items)
                if not allowed: continue
                item_1 = random.choice(allowed)
                batch.append((user_1, item_1))
                batch_items.add(item_1)
                restricted_items.update(set(bought_items))

            batches.extend(batch)

    return batches


def fps_batches_transformer(interactions: dict, batch_size = 32, max_batches_per_user = 10):
    users = list(interactions.keys())
    batches = []
    for user in tqdm(users):
        for _ in range(max_batches_per_user):
            restricted_items = set()
            batch_items = set()
            batch = []

            f_item = random.choice(interactions[user][1:])
            batch.append((user, f_item, interactions[user].index(f_item)))
            batch_items.add(f_item)
            restricted_items.update(set(interactions[user]))

            while len(batch) < batch_size:
                user_1 = random.choice(users)
                if user_1 == user: continue
                bought_items_all = interactions[user_1]
                bought_items = interactions[user_1][1:]
                if not batch_items.isdisjoint(set(bought_items)): continue
                allowed = list(set(bought_items) - restricted_items)
                if not allowed: continue
                item_1 = random.choice(allowed)
                batch.append((user_1, item_1, bought_items_all.index(item_1)))
                batch_items.add(item_1)
                restricted_items.update(set(bought_items))

            batches.extend(batch)

    return batches


def fps_batches_transformer2(interactions: dict, batch_size = 32, max_batches_per_user = 5):
    users = list(interactions.keys())
    batches = []
    for user in tqdm(users):
        for _ in range(max_batches_per_user):
            #restricted_items = set()
            #batch_items = set()
            batch = []

            f_item = random.choice(interactions[user][1:])
            batch.append((user, f_item, interactions[user].index(f_item)))
            user_in_batch = set([user])
            #batch_items.add(f_item)
            #restricted_items.update(set(interactions[user]))

            while len(batch) < batch_size:
                user_1 = random.choice(users)
                if user_1 in user_in_batch: continue
                bought_items_all = interactions[user_1]
                bought_items = interactions[user_1][1:]
                #if not batch_items.isdisjoint(set(bought_items)): continue
                #allowed = list(set(bought_items) - restricted_items)
                #if not allowed: continue
                item_1 = random.choice(bought_items)
                batch.append((user_1, item_1, bought_items_all.index(item_1)))
                user_in_batch.add(user_1)
                #batch_items.add(item_1)
                #restricted_items.update(set(bought_items))

            batches.extend(batch)

    return batches


class FpsLossDataset(torch.utils.data.Dataset):
    def __init__(self, interaction_dataset, user_embeddings, item_embeddings):
        self.interactions = interaction_dataset
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