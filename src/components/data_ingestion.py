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
    raw_data_path: str = "artifacts/raw.csv"
    train_data_path: str = "artifacts/train.csv"
    test_data_path: str = "artifacts/test.csv"


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

            # save the raw data
            self.save_data(data, self.config.raw_data_path)

            return data
        except Exception as e:
            self.logger.error(self.exception_handler(e, sys))
            raise  # re-raise the exception

    # save the raw data as a csv file in the artifacts folder
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

            logging.info("Splitting data into train and test sets")
            train_data, test_data = train_test_split(
                data, test_size=0.2, random_state=42
            )
            logging.info("Data split successfully")

            # save the train and test data
            self.save_data(train_data, self.config.train_data_path)
            self.save_data(test_data, self.config.test_data_path)

            return self.config.train_data_path, self.config.test_data_path
        except Exception as e:
            self.logger.error(self.exception_handler(e, sys))
            raise  # re-raise the exception

    def execute(self) -> None:
        logging.info("Executing data ingestion")
        try:
            logging.info("Reading raw data")
            raw_data = self.read_data("Data/raw.csv")
            logging.info("Raw data read successfully")

            logging.info("Splitting data")
            train_data_path, test_data_path = self.split_data(raw_data)
            logging.info("Data split successfully")

            return train_data_path, test_data_path

        except Exception as e:
            self.logger.error(self.exception_handler(e, sys))
            raise


# if __name__ == "__main__":
#     config = DataIngestionConfig()
#     data_ingestion = DataIngestion(config)
#     train_data_path, test_data_path = data_ingestion.execute()
#     print(train_data_path, test_data_path)

# Path src/components/data_ingestion.py
