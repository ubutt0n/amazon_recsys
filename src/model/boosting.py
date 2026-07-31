from catboost import CatBoostRanker, Pool
from dotenv import load_dotenv
import click
import pandas as pd
import mlflow


@click.command()
@click.argument("dataset_path", type=click.Path())
@click.argument("dataset_val_path", type=click.Path())
@click.argument("experiment_name", type=str)
@click.argument("num_epochs", type=int)
@click.argument("run_name", type=str)
def main(
    dataset_path: str,
    dataset_val_path: str,
    experiment_name: str,
    num_epochs: int,
    run_name: str
) -> None:
    df_f = pd.read_csv(dataset_path)
    df_f_test = pd.read_csv(dataset_val_path)

    text_features = ["cand_title"]
    cat_features = ["cand_main_cat", "cand_sub_cat", "dssm_zone"]

    train_pool = Pool(
        data=df_f.drop(columns=["target", "query_id", "candidate_item_id"]),
        label=df_f["target"],
        group_id=df_f["query_id"],
        cat_features=cat_features,
        text_features=text_features,
    )

    test_pool = Pool(
        data=df_f_test.drop(columns=["target", "query_id", "candidate_item_id"]),
        label=df_f_test["target"],
        group_id=df_f_test["query_id"],
        cat_features=cat_features,
        text_features=text_features,
    )

    model = CatBoostRanker(
        iterations=num_epochs,
        learning_rate=0.05,
        loss_function="YetiRank",
        custom_metric=["NDCG:top=20"],
        eval_metric="NDCG:top=20",
        random_seed=42,
        task_type="GPU",
    
        text_processing={
            "tokenizers": [
                {
                    "tokenizer_id": "Space",
                    "separator_type": "BySense",
                    "lowercasing": "True",
                }
            ],
            "dictionaries": [
                {
                    "dictionary_id": "BiGram",
                    "max_dictionary_size": "50000",
                    "occurrence_lower_bound": "3",
                    "gram_order": "2"
                }
            ],
            "feature_processing": {
                "default": [
                    {
                        "dictionaries_names": ["BiGram"],
                        "feature_calcers": ["BoW"],
                        "tokenizers_names": ["Space"],
                    }
                ]
            },
        }
    )

    mlflow.set_tracking_uri("http://192.168.0.104:5000")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name = run_name):
        model.fit(X=train_pool, eval_set=test_pool, metric_period=10, early_stopping_rounds=50, use_best_model=True)

        val_ndcg = model.evals_result_["validation"]["NDCG:top=20;type=Base"]
        for idx, metric_value in enumerate(val_ndcg):
            step_number = idx * 10
            mlflow.log_metric("ndcg_20_val", metric_value, step=step_number)

        val_pfound = model.evals_result_["validation"]["PFound"]
        for idx, metric_value in enumerate(val_pfound):
            step_number = idx * 10
            mlflow.log_metric("pfound_val", metric_value, step=step_number)

        model.save_model("models/boosting" + "_" + run_name + ".cbm")

        mlflow.log_artifact("models/boosting" + "_" + run_name + ".cbm", artifact_path="model_weights")


if __name__ == "__main__":
    load_dotenv()
    main()