echo "Downloading data"

wget -q https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Grocery_and_Gourmet_Food.jsonl.gz -O "data/raw/Grocery_and_Gourmet_Food.jsonl.gz"
wget -q https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl.gz -O "data/raw/meta_Grocery_and_Gourmet_Food.jsonl.gz"

gunzip data/raw/Grocery_and_Gourmet_Food.jsonl.gz
gunzip data/raw/meta_Grocery_and_Gourmet_Food.jsonl.gz