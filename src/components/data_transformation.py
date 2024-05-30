import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pickle
from src.logger.logging import logging
from src.exception.exception import CustomExceptionHandler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor: object


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.categorical_columns = ["cut", "color", "clarity"]
        self.numerical_columns = ["carat", "depth", "table", "x", "y", "z"]
        self.ordinal_categories = {
            "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
            "color": ["D", "E", "F", "G", "H", "I", "J"],
            "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
        }
        self.logger = logging.getLogger(__name__)
        self.exception_handler = CustomExceptionHandler

    def _get_preprocessor(self) -> object:
        logging.info("Creating preprocessor")
        try:
            numerical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            categories=[
                                self.ordinal_categories["cut"],
                                self.ordinal_categories["color"],
                                self.ordinal_categories["clarity"],
                            ]
                        ),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_transformer, self.numerical_columns),
                    ("cat", categorical_transformer, self.categorical_columns),
                ]
            )
            logging.info("Preprocessor created successfully")

            return preprocessor

        except Exception as e:
            self.logger.error(self.exception_handler(e, sys))
            raise  # re-raise the exception

    # save the preprocessor as a pickle file
    def save_preprocessor(self, preprocessor: object, file_path: str) -> None:
        logging.info("Saving preprocessor")
        try:
            with open(file_path, "wb") as file:
                pickle.dump(preprocessor, file)
            logging.info("Preprocessor saved successfully")
        except Exception as e:
            self.logger.error(self.exception_handler(e, sys))
            raise  # re-raise the exception

    def transform_data(self, data: pd.DataFrame) -> tuple:
        logging.info("Transforming data")
        try:
            logging.info("Getting preprocessor")
            preprocessor = self._get_preprocessor()
            logging.info("Preprocessor obtained successfully")

            logging.info("Transforming data")
            X = data.drop(columns=["id", "price"])
            y = data["price"]

            X_transformed = preprocessor.fit_transform(X)
            logging.info("Data transformed successfully")

            # save the preprocessor
            self.save_preprocessor(preprocessor, "artifacts/preprocessor.pkl")

            return X_transformed, y
        except Exception as e:
            self.logger.error(self.exception_handler(e, sys))
            raise  # re-raise the exception


if __name__ == "__main__":
    config = DataTransformationConfig(preprocessor=None)
    data_transformation = DataTransformation(config)
    data = pd.read_csv("artifacts/train_data.csv")
    # print(data.head(10))
    X, y = data_transformation.transform_data(data)


# Path: src/components/data_transformation.py
