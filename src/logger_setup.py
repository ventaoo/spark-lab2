# src/logger_setup.py
import logging
import os

class AppLogger:
    def __init__(self, name="food_clustering", log_file="./outputs/app.log", level=logging.INFO):
        self.logger = self.setup_logger(name, log_file, level)

    def setup_logger(self, name, log_file, level):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def get_logger(self):
        return self.logger