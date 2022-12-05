import mlflow
if __name__ == "__main__":
   
    with mlflow.start_run(run_name="register_model") as run:
        mlflow.set_tag("mlflow.runName", "register_model")
        model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
        result = mlflow.register_model(model_uri, "trainingmodel-psystock")