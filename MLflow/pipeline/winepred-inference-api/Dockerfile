FROM continuumio/miniconda3
WORKDIR /batch-scoring/
RUN pip install mlflow
ENV MLFLOW_TRACKING_URI=http://localhost:5000
ENTRYPOINT ["mlflow models serve -m "models:/trainingmodel-winepred/Production" -p 6000"]