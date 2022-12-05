import pandas as pd
import mlflow
import mlflow.pyfunc

if __name__ == "__main__":

    with mlflow.start_run(run_name="batch_scoring") as run:

        data=pd.read_csv("data/input.csv", header = None)

        model_name = "training-model-winepred"
        stage = 'Production'

        model = mlflow.pyfunc.load_model(
                model_uri=f"models:/{model_name}/{stage}")

        preds = model.predict(data)
        
        data[len(data.columns)] = preds
        
        result = data

        result.to_csv("data/output.csv")