import pickle
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)

with open("data/processed/items_dssm.pickle", "rb") as f:
    items_dssm = pickle.load(f)

sample_key = list(items_dssm.keys())[0]
VECTOR_DIM = len(items_dssm[sample_key])

COLLECTION_NAME = "dssm_items"

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
)

BATCH_SIZE = 5000
points = []

for int_id, (asin_id, vector) in enumerate(items_dssm.items()):
    
    points.append(
        PointStruct(
            id=int_id,
            vector=np.asarray(vector, dtype=np.float32).tolist(),
            payload={"asin": str(asin_id)} 
        )
    )
    
    if len(points) == BATCH_SIZE:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        points = []
        print(f"  Загружено объектов: {int_id + 1}/{len(items_dssm)}")

if points:
    client.upsert(collection_name=COLLECTION_NAME, points=points)