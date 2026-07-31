import pandas as pd
import numpy as np
from rectools.dataset import Dataset
from rectools import Columns
from rectools.models import ImplicitALSWrapperModel
from implicit.als import AlternatingLeastSquares
import pickle
import click


@click.command()
@click.argument("interactions_input_path", type=click.Path())
@click.argument("items_output", type=click.Path())
@click.argument("users_output", type=click.Path())
@click.argument("als_model_path", type=click.Path())
def get_als_features(
    interactions_input_path: str,
    items_output: str,
    users_output: str,
    als_model_path: str
) -> None:
    
    interaction_dataset = pd.read_csv(interactions_input_path)
    tmp_all = interaction_dataset[["user_id", "parent_asin", "rating", "timestamp"]]
    tmp_all = tmp_all.rename(columns={"user_id": Columns.User, "parent_asin": Columns.Item, "rating": Columns.Weight, "timestamp": Columns.Datetime})
    all_data = Dataset.construct(tmp_all)

    model = ImplicitALSWrapperModel(model=AlternatingLeastSquares())
    model.fit(all_data)

    als_user_embs = {key: model.get_vectors()[0][i] for i, key in enumerate(all_data.user_id_map.external_ids)}
    als_item_embs = {key: model.get_vectors()[1][i] for i, key in enumerate(all_data.item_id_map.external_ids)}

    with open(users_output, "wb") as f: pickle.dump(als_user_embs, f)
    with open(items_output, "wb") as f1: pickle.dump(als_item_embs, f1)

    model.model.save(als_model_path)

if __name__ == "__main__":
    get_als_features()