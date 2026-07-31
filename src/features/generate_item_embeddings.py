import pandas as pd
import numpy as np
from torchvision import models, transforms
import torch
from torch import nn
from tqdm import tqdm
import click
import cv2
import pickle


@click.command()
@click.argument("items_input_path", type=click.Path())
@click.argument("images_input_path", type=click.Path())
@click.argument("items_output_path", type=click.Path())
@click.argument("weights_path", type=click.Path(), default="")
def generate_embeddings(
    items_input_path: str,
    images_input_path: str,
    items_output_path: str,
    weights_path: str,
) -> None:
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    if not weights_path == "":
        model = models.resnet50(weights=None)
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    items = pd.read_csv(items_input_path)
    image_ids = items["parent_asin"].values

    embeddings = {}

    for item in tqdm(image_ids):
        image = cv2.cvtColor(cv2.imread(images_input_path + item + ".png"), cv2.COLOR_BGR2RGB)
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad(): embedding = model(image)
        embeddings[item] = embedding.squeeze().cpu().numpy()
    
    #np.save(items_output_path, np.asarray(embeddings))
    with open(items_output_path, "wb") as f: pickle.dump(embeddings, f)

if __name__ == "__main__":
    generate_embeddings()