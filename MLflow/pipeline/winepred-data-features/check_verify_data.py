import mlflow
from datetime import date
from dateutil.relativedelta import relativedelta
import pprint
import pandas_datareader
import pandas
from pandas_profiling import ProfileReport
import great_expectations as ge
from great_expectations.profile.basic_dataset_profiler import BasicDatasetProfiler


if __name__ == "__main__":
    with mlflow.start_run(run_name="check_verify_data") as run:

        mlflow.set_tag("mlflow.runName", "check_verify_data")

        df = pandas.read_csv("./data/raw/winequality-red.csv")

        describe_to_dict=df.describe().to_dict()
        mlflow.log_dict(describe_to_dict,"describe_data.json")

        pd_df_ge = ge.from_pandas(df)
        
        columns = df.columns
        for column in columns:
            expect = pd_df_ge.expect_column_to_exist(column=column)
            assert expect.success == True

        #we can do some basic cleaning by dropping the null values
        df.dropna(inplace=True)

        #if data_passes_quality_can_go_to_features:
        df.to_csv("data/training/data.csv")