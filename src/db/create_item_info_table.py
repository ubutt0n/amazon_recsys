import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from minio import Minio
from dotenv import load_dotenv
import click
from tqdm import tqdm


@click.command()
@click.argument("items_info_path", type=click.Path())
@click.argument("interactions_path", type=click.Path())
@click.argument("items_image_folder", type=click.Path())
@click.argument("bucket_name", type=str)
def create_item_info_table(items_info_path: str, interactions_path: str, items_image_folder: str, bucket_name: str):
    failed_idx = 239482

    minio_client = Minio(
        os.getenv("MINIO_HOST"),
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        secure=False
    )

    connection_string = f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('DATABASE_HOST')}/{os.getenv('POSTGRES_DB')}"
    engine = create_engine(connection_string)

    items = pd.read_csv(items_info_path)
    items = items[["parent_asin", "title"]]
    inters = pd.read_csv(interactions_path)
    items_count = inters.groupby("parent_asin")["rating"].size()
    
    for idx, row in enumerate(tqdm(items.values)):
        if idx > failed_idx:
            item_id = row[0]
            local_path = os.path.join(items_image_folder, f'{item_id}.png')
            if os.path.exists(local_path):
                minio_path = f"{item_id}.png"
                minio_client.fput_object(bucket_name, minio_path, local_path)
                items.at[idx, 'image_path'] = f"s3://{bucket_name}/{minio_path}"
                items.at[idx, "num_buys"] = items_count[item_id]
            else:
                print(f"Image for {item_id} not found in 'images/' folder")
        else:
            item_id = row[0]
            minio_path = f"{item_id}.png"
            items.at[idx, 'image_path'] = f"s3://{bucket_name}/{minio_path}"
            items.at[idx, "num_buys"] = items_count[item_id]
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS items (
        item_id VARCHAR PRIMARY KEY,
        title TEXT NOT NULL,
        image_path TEXT,
        num_buys INTEGER
    );
    """

    with engine.connect() as conn:
        conn.execute(text(create_table_query))
    
    items.to_sql('items', con=engine, if_exists='replace', index=False)

if __name__ == "__main__":
    load_dotenv()
    create_item_info_table()