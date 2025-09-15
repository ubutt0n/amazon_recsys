from src.model.utils import UserEncoder, ItemEncoder, FpsLossDataset, generate_triplets, TripletDataset, cosine_distance
from src.model.metrics import calculate_map, calculate_ndcg, get_recs_by_batch
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import click
import mlflow
from rectools import Columns
from dotenv import load_dotenv


def run_train_exp_fps_loss(
        #dssm_model: torch.nn.Module,
        item_encoder: torch.nn.Module,
        user_encoder: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        optimizer: torch.optim,
        epoch: int
        #exp_name: str,
        #model_name: str
) -> None:
    
    #mlflow.set_tracking_uri("https://192.168.0.108:5000")
    #mlflow.set_experiment(exp_name)

    user_encoder.train()
    item_encoder.train()
    #dssm_model.train()
    total_loss = 0
    for i, embs in enumerate(tqdm(dataloader)):
        batch_user_feats, batch_item_feats = embs
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
    mlflow.log_metric("clip_loss", avg_loss, step=epoch)
        #print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")


def run_train_exp_triplet_loss(
        item_encoder: torch.nn.Module,
        user_encoder: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        optimizer: torch.optim,
        loss_f: torch.nn,
        epoch: int
) -> None:
    
    user_encoder.train()
    item_encoder.train()
    total_loss = 0
    for embs in tqdm(dataloader):
        anchor_feat, positive_feat, negative_feat = embs
        anchor_feat = anchor_feat.to(device)
        positive_feat = positive_feat.to(device)
        negative_feat = negative_feat.to(device)

        optimizer.zero_grad()

        anchor_emb = user_encoder(anchor_feat)
        positive_emb = item_encoder(positive_feat)
        negative_emb = item_encoder(negative_feat)

        loss = loss_f(anchor_emb, positive_emb, negative_emb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    mlflow.log_metric("triplet_loss", avg_loss, step=epoch)


@click.command()
@click.argument("loss", type=str)
@click.argument("interactions_path", type=click.Path())
@click.argument("interactions_test_path", type=click.Path())
@click.argument("item_embeddings_path", type=click.Path())
@click.argument("user_embeddings_path", type=click.Path())
@click.argument("item_encoder_output_weights", type=click.Path())
@click.argument("user_encoder_output_weights", type=click.Path())
@click.argument("experiment_name", type=str)
@click.argument("num_epochs", type=int)
@click.argument("run_name", type=str)
@click.option("--calc-test-metrics/--no-calc-test-metrics", default=False)
@click.option("--calc-train-metrics/--no-calc-train-metrics", default=False)
def train_dssm(
        loss: str,
        interactions_path: str,
        interactions_test_path: str,
        item_embeddings_path: str,
        user_embeddings_path: str,
        item_encoder_output_weights: str,
        user_encoder_output_weights: str,
        experiment_name: str,
        num_epochs: int,
        run_name: str,
        calc_test_metrics: bool,
        calc_train_metrics: bool
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    interaction_dataset = pd.read_csv(interactions_path)
    interaction_dataset = interaction_dataset[["user_id", "parent_asin", "rating"]]

    with open(item_embeddings_path, "rb") as f: item_embeddings = pickle.load(f)
    with open(user_embeddings_path, "rb") as f: user_embeddings = pickle.load(f)

    interaction_dataset = interaction_dataset[interaction_dataset["parent_asin"].isin(list(item_embeddings.keys()))]

    user_input_dim = 2615
    item_input_dim = 2615
    # old: hidden_dims = [300, 300]
    hidden_dims = [1024, 512, 256]
    embedding_dim = 256

    user_encoder = UserEncoder(user_input_dim, hidden_dims, embedding_dim)
    user_encoder.to(device)

    item_encoder = ItemEncoder(item_input_dim, hidden_dims, embedding_dim)
    item_encoder.to(device)

    #dssm = DSSM(item_encoder, user_encoder)
    #dssm.to(device)
    if loss == "fps_loss":
        dataset = FpsLossDataset(interaction_dataset, user_embeddings, item_embeddings)
        batch_size = 32
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.Adam(list(user_encoder.parameters()) + list(item_encoder.parameters()), lr=1e-3)

        interaction_dataset_test = pd.read_csv(interactions_test_path)
        item_per_user = interaction_dataset_test.groupby("user_id")["rating"].size()
        interaction_dataset_test = interaction_dataset_test[interaction_dataset_test["user_id"].isin(item_per_user[item_per_user < 50].index.values)].reset_index(drop=True)
    else:
        item_per_user = interaction_dataset.groupby("user_id")["rating"].size()
        interaction_dataset_test = interaction_dataset[interaction_dataset["user_id"].isin(item_per_user[item_per_user >= 50].index.values)].reset_index(drop=True)
        interaction_dataset = interaction_dataset[interaction_dataset["user_id"].isin(item_per_user[item_per_user >= 50].index.values)].reset_index(drop=True)
        #dataset = TripletDataset(generate_triplets(interaction_dataset), user_embeddings, item_embeddings)
        #batch_size = 100
        #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        triplet_loss = torch.nn.TripletMarginWithDistanceLoss(margin=0.3, distance_function=cosine_distance)
        optimizer = torch.optim.Adam(list(user_encoder.parameters()) + list(item_encoder.parameters()), lr=1e-3)

    mlflow.set_tracking_uri("http://192.168.0.104:5000")
    mlflow.set_experiment(experiment_name)

    with torch.no_grad():
        user_for_signature = torch.tensor(user_embeddings['AE22236AFRRSMQIKGG7TPTB75QEA']).reshape((1, 2615)).to(device)
        res_for_signature = user_encoder(user_for_signature)
        signature_user = mlflow.models.signature.infer_signature(
            model_input=user_for_signature.cpu().numpy(),
            model_output=res_for_signature.detach().cpu().numpy()
        )

        item_for_signature = torch.tensor(item_embeddings['B0BN7CKZYC']).reshape((1, 2615)).to(device)
        res_for_signature = item_encoder(item_for_signature)
        signature_item = mlflow.models.signature.infer_signature(
            model_input=item_for_signature.cpu().numpy(),
            model_output=res_for_signature.detach().cpu().numpy()
        )


    with mlflow.start_run(run_name = run_name):
        for epoch in range(num_epochs):
            if loss == "fps_loss":
                run_train_exp_fps_loss(item_encoder, user_encoder, dataloader, device, optimizer, epoch)
            else:
                dataset = TripletDataset(generate_triplets(interaction_dataset), user_embeddings, item_embeddings)
                batch_size = 100
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                run_train_exp_triplet_loss(item_encoder, user_encoder, dataloader, device, optimizer, triplet_loss, epoch)

            item_encoder.eval()
            user_encoder.eval()

            with torch.no_grad():
                items_embs = item_encoder(torch.from_numpy(np.asarray(list(item_embeddings.values()))).to(device)).detach().cpu().numpy()
            its = list(item_embeddings.keys())

            if calc_train_metrics:
                inter_data = interaction_dataset[["user_id", "parent_asin", "rating"]].rename(columns={"user_id": Columns.User, "parent_asin": Columns.Item, "rating": Columns.Weight})
                recs_on_train = get_recs_by_batch(
                    interaction_dataset["user_id"].unique(),
                    1000,
                    user_encoder,
                    user_embeddings,
                    device,
                    items_embs,
                    its
                    )
                map_on_train = calculate_map(10, recs_on_train, inter_data)
                ndcg_on_train = calculate_ndcg(10, recs_on_train, inter_data)
                mlflow.log_metric("map_train", map_on_train, step=epoch)
                mlflow.log_metric("ndcg_train", ndcg_on_train, step=epoch)

            if (epoch + 1) % 5 == 0:
                user_encoder.to("cpu")
                mlflow.pytorch.log_model(user_encoder, name = str(loss) + "_user_encoder_model_2", signature=signature_user)
                item_encoder.to("cpu")
                mlflow.pytorch.log_model(item_encoder, name = str(loss) + "_item_encoder_model_2", signature=signature_item)

                user_encoder.to(device)
                item_encoder.to(device)
                
                if calc_test_metrics:
                    inter_data = interaction_dataset_test[["user_id", "parent_asin", "rating"]].rename(columns={"user_id": Columns.User, "parent_asin": Columns.Item, "rating": Columns.Weight})
                    inter_data = inter_data.loc[inter_data["user_id"].isin(interaction_dataset_test["user_id"].unique()[0:10000])]
                    recs_on_test = get_recs_by_batch(
                        interaction_dataset_test["user_id"].unique()[0:10000],
                        1000,
                        user_encoder,
                        user_embeddings,
                        device,
                        items_embs,
                        its
                        )
                    map_on_test = calculate_map(10, recs_on_test, inter_data)
                    ndcg_on_test = calculate_ndcg(10, recs_on_test, inter_data)
                    mlflow.log_metric("map_test", map_on_test, step=epoch)
                    mlflow.log_metric("ndcg_test", ndcg_on_test, step=epoch)

            torch.save(item_encoder.state_dict(), item_encoder_output_weights + "_" + str(epoch) + ".pth")
            torch.save(user_encoder.state_dict(), user_encoder_output_weights + "_" + str(epoch) + ".pth")

            mlflow.log_artifact(user_encoder_output_weights + "_" + str(epoch) + ".pth", artifact_path="model_weights")
            mlflow.log_artifact(item_encoder_output_weights + "_" + str(epoch) + ".pth", artifact_path="model_weights")



if __name__ == "__main__":
    load_dotenv()
    train_dssm()