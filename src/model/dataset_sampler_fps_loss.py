import numpy as np
import pandas as pd
from tqdm import tqdm
import click


@click.command()
@click.argument("interactions_input_path", type=click.Path())
@click.argument("sampled_dataset_output_path", type=click.Path())
@click.argument("batch_size", type=int)
@click.argument("min_interactions_num", type=int)
def sample_dataset(
    interactions_input_path: str,
    sampled_dataset_output_path: str,
    batch_size: int,
    min_interactions_num: int
) -> None:

    interaction_dataset = pd.read_csv(interactions_input_path)
    interaction_dataset = interaction_dataset.drop_duplicates(subset=["user_id", "parent_asin"]).reset_index(drop=True)
    item_per_user = interaction_dataset.groupby("user_id")["rating"].size()
    interaction_dataset = interaction_dataset[interaction_dataset["user_id"].isin(item_per_user[item_per_user >= min_interactions_num].index.values)].reset_index(drop=True)
    interaction_dataset["block"] = None
    interaction_dataset["user_sort"] = pd.Categorical(interaction_dataset["user_id"], categories=interaction_dataset.groupby('user_id').size().sort_values(ascending=False).index, ordered=True)
    interaction_dataset = interaction_dataset.sort_values("user_sort").reset_index(drop=True).drop("user_sort", axis=1)

    grouped = interaction_dataset.groupby("user_id", sort=False)
    def gen_batch(batch_n):
        restricted_items = set()
        batch_indexes = []
        batch_items = set()

        for _, group in grouped:
            if len(batch_indexes) == batch_size: break
            bought_items = group[["parent_asin", "block"]].values
            if not batch_items.isdisjoint(set(bought_items[:, 0])): continue
            choosed_item = None
            for item, block in bought_items:
                if choosed_item is None and item not in restricted_items and block is None:
                    inter_index = group[group["parent_asin"] == item].index.values[0]
                    batch_indexes.append(inter_index)
                    batch_items.add(item)
                    choosed_item = item
                    break
            if choosed_item is not None:
                restricted_items.update(set(bought_items[:, 0]))
        interaction_dataset.loc[batch_indexes, "block"] = batch_n

        return 1 if len(batch_indexes) == 32 else 0
    
    num_batches = int(interaction_dataset.shape[0] / batch_size) + 1

    for i in tqdm(range(num_batches)):
        batched = gen_batch(i)
        if not batched: break

    inter_sampled = interaction_dataset[interaction_dataset["block"].isin(interaction_dataset["block"].dropna())].sort_values(by="block")

    inter_sampled.to_csv(sampled_dataset_output_path, index=False)

if __name__ == "__main__":
    sample_dataset()