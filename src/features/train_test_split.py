import pandas as pd
import numpy as np
import click


@click.command()
@click.argument("interactions_input_path", type=click.Path())
@click.argument("interactions_train_path", type=click.Path())
@click.argument("interactions_test_path", type=click.Path())
def train_test_split(
    interactions_input_path: str,
    interactions_train_path: str,
    interactions_test_path: str
) -> None:
    
    interaction_dataset = pd.read_csv(interactions_input_path)

    ratings_per_user = interaction_dataset.groupby('user_id')['rating'].count().sort_values(ascending=False)
    remove_users = []

    for user_id, num_ratings in ratings_per_user.items():
        if num_ratings < 5:
            remove_users.append(user_id)

    interaction_dataset = interaction_dataset.loc[ ~ interaction_dataset['user_id'].isin(remove_users)]

    test_dataset = interaction_dataset.sort_values(['user_id', 'timestamp'], ascending=[True, False]).groupby("user_id").head(2)
    train_dataset = interaction_dataset.drop(test_dataset.index.values)

    train_dataset.to_csv(interactions_train_path, index=False)
    test_dataset.to_csv(interactions_test_path, index=False)

if __name__ == "__main__":
    train_test_split()