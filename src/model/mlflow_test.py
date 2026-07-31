import os
from random import random, randint
from mlflow import log_artifacts, log_param, log_metric
import mlflow
import boto3
import os
from dotenv import load_dotenv

load_dotenv()


mlflow.set_tracking_uri("http://192.168.0.108:5000")
mlflow.set_experiment("mlflow_test")

if __name__ == "__main__":
    log_param("param1", randint(0, 100))

    log_metric("m", random())
    log_metric("m", random() + 1)
    log_metric("m", random() + 2)

    if not os.path.exists("test_outs"): os.makedirs("test_outs")
    with open("test_outs/mlflow_test.txt", "w") as f: f.write("test")
    log_artifacts("test_outs")