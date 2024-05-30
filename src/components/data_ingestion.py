import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger.logging import logging
from src.exception.exception import CustomExceptionHandler


@dataclass
class DataIngestionConfig:
    train_data_path: str
    test_data_path: str


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.exception_handler = CustomExceptionHandler

    def read_data(self, data_path: str) -> pd.DataFrame:
        logging.info("Reading data")
        try:
            data = pd.read_csv(data_path, low_memory=False)
            logging.info("Data read successfully")
            return data
        except Exception as e:
            self.logger.error(self.exception_handler(e, sys))
            raise  # re-raise the exception

    def save_data(self, data: pd.DataFrame, file_path: str) -> None:
        logging.info("Saving data")
        try:
            data.to_csv(file_path, index=False)
            logging.info("Data saved successfully")
        except Exception as e:
            self.logger.error(self.exception_handler(e, sys))
            raise  # re-raise the exception

    def split_data(self, data: pd.DataFrame) -> None:
        logging.info("Splitting data")
        try:
            train_data, test_data = train_test_split(
                data, test_size=0.2, random_state=42
            )
            self.save_data(train_data, self.config.train_data_path)
            self.save_data(test_data, self.config.test_data_path)
            logging.info("Data split successfully")
        except Exception as e:
            self.logger.error(self.exception_handler(e, sys))
            raise  # re-raise the exception


if __name__ == "__main__":
    config = DataIngestionConfig(
        train_data_path="artifacts/train_data.csv",
        test_data_path="artifacts/test_data.csv",
    )
    data_ingestion = DataIngestion(config)
    data = data_ingestion.read_data("Data/raw.csv")
    data_ingestion.split_data(data)
