import logging
import os
import click
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

_steps = [
    "data_features",
    "training",
    "inference_batch"
]


@click.command()
@click.option("--steps", default="all", type=str)
def run_pipeline(steps):

    EXPERIMENT_NAME = "winepred/exp_name"
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    logger.info("pipeline experiment_id: %s", experiment.experiment_id)
    
    # Steps to execute
    active_steps = steps.split(",") if steps != "all" else _steps
    logger.info("pipeline active steps to execute in this run: %s", active_steps)

    with mlflow.start_run(run_name='pipeline', nested=True) as active_run:
        if "data_features" in active_steps:
            data_features_run = mlflow.run(".", "data_features", parameters={})
            data_features_run = mlflow.tracking.MlflowClient().get_run(data_features_run.run_id)
            logger.info(data_features_run)

        if "training" in active_steps:
            training_run = mlflow.run(".", "training", parameters={})
            training_run = mlflow.tracking.MlflowClient().get_run(training_run.run_id)
            model_uri = os.path.join(train_run.info.artifact_uri,"model")
            mlflow.register_model(model_uri,"training-model-winepred")
            logger.info(training_run)
            
        if "inference_batch" in active_steps:
            inference_batch_run = mlflow.run(".", "inference_batch", parameters={})
            inference_batch_run = mlflow.tracking.MlflowClient().get_run(inference_batch_run.run_id)
            logger.info(inference_batch_run)
        
        if "inference_api" in active_steps:
            inference_api_run = mlflow.run(".", "inference_api", parameters={})
            inference_api_run = mlflow.tracking.MlflowClient().get_run(inference_api_run.run_id)
            logger.info(inference_api_run)

    logger.info('finished mlflow pipeline run with a run_id = %s', active_run.info.run_id)

if __name__ == "__main__":
    run_pipeline()