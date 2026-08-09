from datetime import timedelta
from feast import Entity, FeatureView, Field, ValueType
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.types import Int64, Float32, String


candidate_item_id = Entity(name="candidate_item_id", value_type=ValueType.STRING)

item_postgres_source = PostgreSQLSource(
    name="item_features_source",
    table="items_feast_table",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

item_features_view = FeatureView(
    name="item_features",
    entities=[candidate_item_id],
    ttl=timedelta(days=365),
    schema=[
        Field(name="cand_main_cat", dtype=String),
        Field(name="cand_sub_cat", dtype=String),
        Field(name="cand_amazon_rating", dtype=Float32),
        Field(name="cand_amazon_rat_num", dtype=Int64),
        Field(name="cand_title", dtype=String),
        Field(name="transformer_input_vector", dtype=String),
    ],
    online=True,
    source=item_postgres_source,
)