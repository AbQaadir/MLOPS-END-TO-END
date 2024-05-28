import pandas as pd
import numpy as np
import os
import sys

from src.logger.logging import logging
from src.exception.exception import CustomExceptionHandler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    train_data_path: str
    test_data_path: str

    def __post_init__(self):
        self.train_data_path = Path(self.train_data_path)
        self.test_data_path = Path(self.test_data_path)

        if not self.train_data_path.exists():
            raise FileNotFoundError(f"File not found: {self.train_data_path}")
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"File not found: {self.test_data_path}")

        if not self.train_data_path.is_file():
            raise FileNotFoundError(f"Invalid file: {self.train_data_path}")
        if not self.test_data_path.is_file():
            raise FileNotFoundError(f"Invalid file: {self.test_data_path}")

        if not self.train_data_path.suffix == ".csv":
            raise ValueError(f"Invalid file format: {self.train_data_path}")
        if not self.test_data_path.suffix == ".csv":
            raise ValueError(f"Invalid file format: {self.test_data_path}")

        if not self.train_data_path.stat().st_size > 0:
            raise ValueError(f"Empty file: {self.train_data_path}")
        if not self.test_data_path.stat().st_size > 0:
            raise ValueError(f"Empty file: {self.test_data_path}")

    def __str__(self):
        return f"train_data_path: {self.train_data_path}\ntest_data_path: {self.test_data_path}"

    def __repr__(self):
        return f"train_data_path: {self.train_data_path}\ntest_data_path: {self.test_data_path}"


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.logger = get_logger()
        self.exception_handler = CustomExceptionHandler(logger=self.logger)

    def read_data(self):
        try:
            self.logger.info("Reading data from source")
            self.train_data = pd.read_csv(self.config.train_data_path)
            self.test_data = pd.read_csv(self.config.test_data_path)
            self.logger.info("Data read successfully")
        except Exception as e:
            self.exception_handler.handle_exception(e)
            sys.exit(1)

    def split_data(self):
        try:
            self.logger.info("Splitting data into train and test")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.train_data.drop("target", axis=1),
                self.train_data["target"],
                test_size=0.2,
                random_state=42,
            )
            self.logger.info("Data split successfully")
        except Exception as e:
            self.exception_handler.handle_exception(e)
            sys.exit(1)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test



if __name__ == "__main__":
    
    config = DataIngestionConfig(
        train_data_path="data/train.csv",
        test_data_path="data/test.csv"
    )
    data_ingestion = DataIngestion(config)
    data_ingestion.read_data()
    data_ingestion.split_data()
    X_train, y_train = data_ingestion.get_train_data()
    X_test, y_test = data_ingestion.get_test_data()
    print(X_train.head())
    print(y_train.head())
    print(X_test.head())
    print(y_test.head())
    
    
# Path src/components/data_ingestion.py