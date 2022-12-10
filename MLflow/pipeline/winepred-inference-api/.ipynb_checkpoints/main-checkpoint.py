import mlflow
import click
import os

def _run(entrypoint, parameters={}, source_version=None, use_cache=True):
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


@click.command()
def workflow():
    with mlflow.start_run(run_name ="pipeline") as active_run:
        mlflow.set_tag("mlflow.runName", "pipeline")
        pipeline_model_run = _run("inference_pipeline_model")     
        
        
if __name__=="__main__":
    workflow()