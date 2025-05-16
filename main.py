import sys
import os
from src.spark_manager import SparkManager
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.visualizer import Visualizer
from src.logger_setup import AppLogger

def main():
    # 解析命令行参数
    config_path, data_path, model_path = sys.argv[1:4]
    os.makedirs("./outputs/spark-events", exist_ok=True)

    # 初始化日志
    logger = AppLogger().get_logger()
    logger.info("程序启动，开始加载配置")

    # 初始化 Spark 管理器
    spark_manager = SparkManager(config_path)
    spark = spark_manager.spark
    logger.info("Spark session 创建成功...")

    # 定义列名和有效范围
    COLUMNS = ["fat_100g", "carbohydrates_100g", "proteins_100g"]
    RANGES = {
        "fat_100g": (0, 100),
        "carbohydrates_100g": (0, 100),
        "proteins_100g": (0, 100),
    }

    # 数据处理
    data_processor = DataProcessor(spark)
    df_clean = data_processor.load_and_clean_data(data_path, COLUMNS, RANGES)
    df_scaled = data_processor.assemble_and_scale(df_clean, COLUMNS)
    df_sampled = df_scaled.sample(fraction=0.1, seed=42)

    # 模型训练
    logger.info('模型开始训练...')
    model_trainer = ModelTrainer(k=3, seed=42)
    model_trainer.train_kmeans(df_sampled)
    predictions = model_trainer.model.transform(df_sampled)
    logger.info(f"Inertia (训练成本): {model_trainer.model.summary.trainingCost:.2f}")

    # 保存模型
    try:
        if os.path.exists(model_path):
            import datetime
            model_path = f"{model_path}_{datetime.datetime.now().strftime('%H:%M:%S')}"
        model_trainer.save_model(model_path)
        logger.info(f"模型保存成功: {os.path.abspath(model_path)}")
    except Exception as e:
        logger.error(f"保存模型失败: {e}")

    # 可视化结果
    predictions_pdf = predictions.toPandas()
    visualizer = Visualizer()

    pca_plot_path = visualizer.plot_pca_clusters(predictions_pdf)
    count_plot_path = visualizer.plot_cluster_counts(predictions_pdf)

    logger.info(f"PCA 聚类图已保存至: {pca_plot_path}")
    logger.info(f"聚类数量分布图已保存至: {count_plot_path}")

if __name__ == "__main__":
    main()