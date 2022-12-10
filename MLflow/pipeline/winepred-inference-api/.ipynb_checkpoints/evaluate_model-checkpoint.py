import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def classification_metrics(df:None):
    metrics={}
    metrics["mae"]=mean_absolute_error(df["preds"], df["y_test"]  )
    return metrics
    
if __name__ == "__main__":

    with mlflow.start_run(run_name="evaluate_model") as run:
        mlflow.set_tag("mlflow.runName", "evaluate_model")
        df=pd.read_csv("data/predictions/test_predictions.csv")
        metrics = classification_metrics(df)
        mlflow.log_metrics(metrics)