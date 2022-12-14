import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

spark = SparkSession.builder.appName("Batch inference with MLflow DL inference pipeline").getOrCreate()

logged_model = 'models:/inference_pipeline_model/1'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='float')

# Predict on a Spark DataFrame.
df = spark.read.csv('pipeline/data/input.csv', header=True)
df = df.withColumn('predictions', loaded_model())

df.show(n = 10, truncate=80, vertical=True)