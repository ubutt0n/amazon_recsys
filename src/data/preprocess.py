import pandas as pd
import json
import click
import os


@click.command()
@click.argument("items_input_path", type=click.Path())
@click.argument("interactions_input_path", type=click.Path())
@click.argument("items_output_path", type=click.Path())
@click.argument("interactions_output_path", type=click.Path())
@click.argument("images_path", type=click.Path())
def preprocess(
    items_input_path: str,
    interactions_input_path: str,
    items_output_path: str,
    interactions_output_path: str,
    images_path: str
) -> None:
    
    item_keys = ['title', 'average_rating', 'rating_number', 'images', 'categories', 'parent_asin']
    item_data = []

    with open(items_input_path, 'r') as fp:
        for line in fp:
            t = json.loads(line.strip())
            item = [t[key] for key in item_keys]
            if len(item[3]) == 0: continue
            item[3] = (item[3][0])["large"]
            item[4] = ", ".join(item[4])
            item_data.append(item)
    
    item_dataset = pd.DataFrame(data=item_data, columns=item_keys)

    inter_keys = ['rating', 'parent_asin', 'user_id', 'timestamp', 'verified_purchase']
    inter_data = []

    with open(interactions_input_path, 'r') as fp:
        for line in fp:
            t = json.loads(line.strip())
            inter_data.append([t[key] for key in inter_keys])

    inter_dataset = pd.DataFrame(data=inter_data, columns=inter_keys)

    # remove items with less then 5 reviews
    ratings_per_item = inter_dataset.groupby('parent_asin')['rating'].count().sort_values(ascending=False)
    remove_items = []

    for item_id, num_ratings in ratings_per_item.items():
        if num_ratings < 5:
            remove_items.append(item_id)
    
    item_dataset = item_dataset.loc[ ~ item_dataset['parent_asin'].isin(remove_items)]
    inter_dataset = inter_dataset.loc[ ~ inter_dataset['parent_asin'].isin(remove_items)]

    # remove items without images and categories
    image_not_exist = []
    for i in item_dataset["parent_asin"].values:
        if not os.path.exists(images_path + i + ".png"):
            image_not_exist.append(i)
    item_dataset = item_dataset.loc[ ~ item_dataset['parent_asin'].isin(image_not_exist)]
    inter_dataset = inter_dataset.loc[ ~ inter_dataset['parent_asin'].isin(image_not_exist)]

    items_without_categories = item_dataset.loc[item_dataset["categories"] == ""]["parent_asin"].values
    item_dataset = item_dataset.loc[ ~ item_dataset['parent_asin'].isin(items_without_categories)]
    inter_dataset = inter_dataset.loc[ ~ inter_dataset['parent_asin'].isin(items_without_categories)]

    shared_items = set(inter_dataset["parent_asin"].unique()) & set(item_dataset["parent_asin"].unique())
    item_dataset = item_dataset.loc[item_dataset["parent_asin"].isin(shared_items)]
    inter_dataset = inter_dataset.loc[inter_dataset["parent_asin"].isin(shared_items)]

    item_dataset.to_csv(items_output_path, index=False)
    inter_dataset.to_csv(interactions_output_path, index=False)


if __name__ == "__main__":
    preprocess()