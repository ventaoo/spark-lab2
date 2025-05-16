# src/model_trainer.py
from pyspark.ml.clustering import KMeans

class ModelTrainer:
    def __init__(self, k=3, seed=42):
        self.k = k
        self.seed = seed

    def train_kmeans(self, df):
        kmeans = KMeans(k=self.k, featuresCol="scaled_features", seed=self.seed)
        self.model = kmeans.fit(df)

    def save_model(self, path: str):
        self.model.save(path)

    def get_model(self):
        return self.model