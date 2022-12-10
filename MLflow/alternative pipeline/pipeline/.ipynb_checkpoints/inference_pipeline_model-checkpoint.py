# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import mlflow
import pandas as pd
from pathlib import Path


# %%
class InferencePipeline(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.finetuned_model = mlflow.sklearn.load_model(self.finetuned_model_uri)
    
    def __init__(self, finetuned_model_uri):
        self.finetuned_model_uri = finetuned_model_uri
    
    def predict(self, context, model_input):
        results = self.finetuned_model.predict(model_input)
        return results



# %%
CONDA_ENV = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent, "conda.yaml")


# %%
@click.command(help="This program creates a multi-step inference pipeline model .")
@click.option("--finetuned_model_run_id", default=None)

df=pd.read_csv("data/training/data.csv")

MODEL_ARTIFACT_PATH = 'inference_pipeline_model'
with mlflow.start_run(run_name="inference_pipeline") as model_tracking_run:
    
    finetuned_model_uri = f'runs:/{finetuned_model_run_id}/model'
    inference_pipeline_uri = f'runs:/{model_tracking_run.info.run_id}/{MODEL_ARTIFACT_PATH}'
    mlflow.pyfunc.log_model(artifact_path=MODEL_ARTIFACT_PATH, 
                            conda_env=CONDA_ENV, 
                            python_model=InferencePipeline(finetuned_model_uri))     