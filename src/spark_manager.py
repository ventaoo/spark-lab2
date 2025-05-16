# src/spark_manager.py
from pyspark.sql import SparkSession

class SparkManager:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.spark = self.create_spark_session()

    def load_config(self, path: str) -> dict:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def create_spark_session(self) -> SparkSession:
        builder = SparkSession.builder.appName(
            self.config.get("app_name", "MySparkApp")
        )
        for k, v in self.config.get("config", {}).items():
            builder = builder.config(k, v)
        return builder.getOrCreate()