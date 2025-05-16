# src/data_processor.py
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler

class DataProcessor:
    def __init__(self, spark):
        self.spark = spark

    def load_and_clean_data(self, path: str, selected_columns: list, valid_ranges: dict) -> DataFrame:
        df = self.spark.read.option("sep", "\t").option("header", True).option("inferSchema", True).csv(path)
        df = df.select([col(c).cast("float") for c in selected_columns]).na.drop()
        for column, (lower, upper) in valid_ranges.items():
            df = df.withColumn(column, when((col(column) >= lower) & (col(column) <= upper), col(column)).otherwise(None))
        return df.na.drop()

    def assemble_and_scale(self, df: DataFrame, input_cols: list) -> DataFrame:
        assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
        df_vec = assembler.transform(df)
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
        scaler_model = scaler.fit(df_vec)
        return scaler_model.transform(df_vec)