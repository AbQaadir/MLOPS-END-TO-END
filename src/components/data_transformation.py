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
    transformed_X_train_data_path: str = "artifacts/transformed/X_train.csv"
    transformed_y_train_data_path: str = "artifacts/transformed/y_train.csv"
    transformed_X_test_data_path: str = "artifacts/transformed/X_test.csv"
    transformed_y_test_data_path: str = "artifacts/transformed/y_test.csv"
    preprocessor_path: str = "artifacts/preprocessor.pkl"


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

    def transform_data(self, train_data_path: str, test_data_path: str):
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)

        X_train = train_df.drop(columns=["id", "price"])
        y_train = train_df["price"]

        X_test = test_df.drop(columns=["id", "price"])
        y_test = test_df["price"]

        preprocessor = self._get_preprocessor()

        logging.info("Fitting preprocessor")

        try:
            logging.info("Fitting preprocessor")
            transormed_X_train = preprocessor.fit_transform(X_train)
            transformed_X_test = preprocessor.transform(X_test)
            logging.info("Preprocessor fitted successfully")

            logging.info("Saving preprocessor")
            self.save_preprocessor(preprocessor, self.config.preprocessor_path)
            logging.info("Preprocessor saved successfully")

            train_arr = np.c_[transormed_X_train, np.array(y_train)]
            test_arr = np.c_[transformed_X_test, np.array(y_test)]

            return train_arr, test_arr

        except Exception as e:
            self.logger.error(self.exception_handler(e, sys))
            raise  # re-raise the exception


# if __name__ == "__main__":
#     config = DataTransformationConfig()
#     data_transformation = DataTransformation(config)
#     train_data_path = "artifacts/train.csv"
#     test_data_path = "artifacts/test.csv"
#     train_arr, test_arr = data_transformation.transform_data(train_data_path, test_data_path)

# Path: src/components/data_transformation.py
