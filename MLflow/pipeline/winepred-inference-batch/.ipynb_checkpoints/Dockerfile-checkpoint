FROM continuumio/miniconda3

WORKDIR /batch-scoring/

RUN pip install mlflow

RUN pip install pandas

COPY batch_scoring.py   /batch-scoring/
COPY MLproject          /batch-scoring/

ENV MLFLOW_TRACKING_URI=http://localhost:5000

ENTRYPOINT ["mlflow run . --no-conda"]