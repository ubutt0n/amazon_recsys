from src.backend.tools import Model, UserData, get_sql_engine, get_minio_client
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
from io import BytesIO
from sqlalchemy import text


als_w = "models/als_model.npz"
iidm = "data/processed/item_id_map.json"
item_embs = "data/processed/items_full_CLIP_wo_als.pickle"
item_enc = "data/processed/items_full_encoded_test.pickle"
with open(iidm, "r") as jsfile: iid_map = json.load(jsfile)

model = Model("serve_test_user", als_w, iidm, item_embs, item_enc)

sql_engine = get_sql_engine()
minio_client = get_minio_client()

app = FastAPI()

@app.get("/get_item_ids")
async def get_item_ids():
    return list(iid_map.keys())

@app.get("/get_item_image")
async def get_item_image(item_id: str):
    image_stream = BytesIO(minio_client.get_object("images", item_id + ".png").read())
    return StreamingResponse(image_stream, media_type="application/octet-stream")

@app.get("/get_item_title")
async def get_item_name(item_id: str):
    conn = sql_engine.connect()
    query = text(f"""SELECT title FROM public.items WHERE parent_asin = '{item_id}'""")
    result = conn.execute(query).fetchall()[0][0]
    return result

@app.get("/get_popular_items")
async def get_popular_items(num_items: int):
    conn = sql_engine.connect()
    query = text(f"""SELECT parent_asin FROM public.items ORDER BY num_buys DESC LIMIT {num_items};""")
    result = conn.execute(query).fetchall()
    return list(map(lambda x: x[0], result))

@app.post("/recommend_with_model")
async def recommend_with_model(
    user_data: UserData, num_recs: int = 10
):
    recs = model.recommend(user_data, num_recs)
    return recs