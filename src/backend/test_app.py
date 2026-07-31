from src.backend.tools import Model, UserData
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from PIL import Image


if __name__ == "__main__":
    load_dotenv()

    m = Model("serve_test_user", "models/als_model.npz", "data/processed/item_id_map.json", "data/processed/items_full_CLIP_wo_als.pickle", "data/processed/items_full_encoded_test.pickle")
    items = ['B0BN7CKZYC', 'B0C2W77WJX', 'B07TTJYHGX', 'B0BYLK9168']
    ranks = [5, 2, 4, 5]
    u = UserData()
    u.items = items
    u.ranks = ranks

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    for ax, im in zip(axes, items):
        image = Image.open("data/interim/item_images/" + im + ".png").convert("RGB")
        ax.imshow(image)

    recs = m.recommend(u, 9)

    fig1, axes1 = plt.subplots(3, 3, figsize=(9, 9))
    axes1 = axes1.flatten()
    for ax, im in zip(axes1, recs):
        image = Image.open("data/interim/item_images/" + im + ".png").convert("RGB")
        ax.imshow(image)
    plt.show()