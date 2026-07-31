import pandas as pd
import click
import requests
import numpy as np
import cv2
import os


@click.command()
@click.argument("items_dataset_input_path", type=click.Path())
@click.argument("items_images_output_path", type=click.Path())
def get_images(items_dataset_input_path: str, items_images_output_path: str) -> None:

    items_dataset = pd.read_csv(items_dataset_input_path)

    os.mkdir(items_images_output_path)

    for _, item in items_dataset.iterrows():
        image_url = item.iloc[3]
        name = item.iloc[5]
        item_image = cv2.imdecode(np.asarray(bytearray(requests.get(image_url).content), np.uint8), cv2.IMREAD_COLOR)
        if item_image is None: continue
        cv2.imwrite(os.path.join(items_images_output_path, name + ".png"), cv2.resize(item_image, (256,256)))
    

if __name__ == "__main__":
    get_images()