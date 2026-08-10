from src.backend.tools import get_sql_engine, get_minio_client, ModelTransformer, UserDataT
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import json
from sqlalchemy import text
from functools import lru_cache


iidm = "item_id_map.json"
with open(iidm, "r") as jsfile: iid_map = json.load(jsfile)

model_ = ModelTransformer()

sql_engine = get_sql_engine()
minio_client = get_minio_client()

app = FastAPI()

@lru_cache(maxsize=1024)
def fetch_from_minio(item_id: str) -> bytes:
    try:
        response = minio_client.get_object("images", f"{item_id}.png")
        data = response.read()
        response.close()
        response.release_conn()
        return data
    except Exception as e:
        raise HTTPException(status_code=404, detail="Image not found")

@app.get("/get_item_ids")
def get_item_ids():
    return list(iid_map.keys())

@app.get("/get_item_image")
def get_item_image(item_id: str):
    #image_stream = BytesIO(minio_client.get_object("images", item_id + ".png").read())
    #return StreamingResponse(image_stream, media_type="application/octet-stream")
    image_bytes = fetch_from_minio(item_id)
    return Response(content=image_bytes, media_type="image/png")

@app.get("/get_item_title")
def get_item_name(item_id: str):
    with sql_engine.connect() as conn:
        query = text(f"""SELECT title FROM public.items WHERE parent_asin = '{item_id}'""")
        result = conn.execute(query).fetchall()[0][0]
        return result

@app.get("/get_popular_items")
def get_popular_items(num_items: int):
    conn = sql_engine.connect()
    query = text(f"""SELECT parent_asin FROM public.items ORDER BY num_buys DESC LIMIT {num_items};""")
    result = conn.execute(query).fetchall()
    return list(map(lambda x: x[0], result))

@app.post("/recommend_with_model")
def recommend_with_model(
    user_data: UserDataT, num_recs: int = 10
):
    recs = model_.recommend(user_data, num_recs)
    print(recs)
    return recs