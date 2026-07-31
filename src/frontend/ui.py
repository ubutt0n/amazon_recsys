from typing import List
import requests
import streamlit as st
from io import BytesIO
import random


def get_recs(
    item_ids: List[str],
    num_recs: int
) -> List[int]:
    user_data = {"items": item_ids}
    payload = {
        "num_recs": num_recs,
    }
    result_json = requests.post(
        f"http://model_service:8003/recommend_with_model/",
        json=user_data,
        timeout=300,
        params=payload,
    ).json()
    print(result_json)
    return result_json

@st.cache_data
def get_item_image(item_id: str):
    item_data = {"item_id": item_id}
    image_stream = requests.get("http://model_service:8003/get_item_image/", params=item_data).content
    return BytesIO(image_stream)

@st.cache_data
def get_item_title(item_id: str) -> str:
    item_data = {"item_id": item_id}
    title = requests.get("http://model_service:8003/get_item_title/", params=item_data).json()
    return title

@st.cache_data
def get_popular_items(num_items: int):
    item_data = {"num_items": num_items}
    popular_items = requests.get("http://model_service:8003/get_popular_items/", params=item_data).json()
    return popular_items


st.title("Рекомендательная система на датасете Amazon Reviews")

if "items" not in st.session_state:
    st.session_state["items"] = random.sample(get_popular_items(100), 20)
    st.session_state["ratings"] = [0] * len(st.session_state["items"])

st.write("Оцените товары от 1 до 5:")

cols_per_row = 4

for row_start in range(0, len(st.session_state["items"]), cols_per_row):
    row_items = list(st.session_state["items"])[row_start : row_start + cols_per_row]
    #row_items = st.session_state["items"][row_start:row_start + cols_per_row]
    cols = st.columns(len(row_items))

    for col, item_id, idx in zip(cols, row_items, range(row_start, row_start + len(row_items))):
        with col:
            st.image(get_item_image(item_id), width="stretch")
            st.write(get_item_title(item_id))
            st.session_state["ratings"][idx] = st.slider(
                f"Оценка для {item_id}", 1, 5, value=st.session_state["ratings"][idx], key=f"slider_{item_id}"
            )

filtered_items = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
if st.button("Рекомендовать"):

    #filtered_ratings = []
    for item, rating in zip(st.session_state["items"], st.session_state["ratings"]):
        if rating > 0:
            filtered_items.append(item)
            #filtered_ratings.append(rating)
    if filtered_items:
        new_items = get_recs(filtered_items[len(filtered_items)-10:], 20)
        st.session_state["items"] = new_items
        st.session_state["ratings"] = [0] * len(new_items)

    st.rerun()