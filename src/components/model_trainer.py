import pandas as pd
import numpy as np
import pickle
import os
import sys

from src.utils.utils import Utils
from src.exception.exception import CustomExceptionHandler
from src.logger.logging import logging
from dataclasses import dataclass

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV


@dataclass
class ModelTrainerConfig:
    model_path: str


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.exception_handler = CustomExceptionHandler
        self.logger = logging.getLogger(__name__)
        self.models = {
            "LinearRegression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "Elasticnet": ElasticNet(),
            "Randomforest": RandomForestRegressor(),
            "xgboost": XGBRegressor(),
        }
        self.param_grid = {
            "Lasso": {"regressor__alpha": [0.1, 1.0, 10.0]},
            "Ridge": {"regressor__alpha": [0.1, 1.0, 10.0]},
            "Elasticnet": {
                "regressor__alpha": [0.1, 1.0, 10.0],
                "regressor__l1_ratio": [0.1, 0.5, 0.9],
            },
            "Randomforest": {"regressor__n_estimators": [50, 100, 200]},
            "xgboost": {
                "regressor__n_estimators": [50, 100, 200],
                "regressor__max_depth": [3, 5, 7],
            },
        }
        self.utils = Utils()

    def train_model(self, model_name, data_path):
        try:
            # load the data
            logging.info("Loading data")
            data = self.utils.load_data(data_path)
            logging.info("Data loaded successfully")

            # split the data
            logging.info("Splitting data")
            X_train, y_train = self.utils.data_split(data)
            logging.info("Data split successfully")

            # load the model
            logging.info(f"Training {model_name} model")
            model = self.models[model_name]
            logging.info(f"Model {model_name} loaded successfully")

            # load the preprocessor
            logging.info("Loading preprocessor")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            preprocessor = self.utils.load_preprocessor(preprocessor_path)
            logging.info("Preprocessor loaded successfully")

            # make a model pipeline
            logging.info("Making model pipeline")
            model_pipeline = self.utils.make_pipeline(preprocessor = preprocessor, model = model)
            logging.info("Model pipeline created successfully")

            # hyperparameter tuning
            logging.info("Hyperparameter tuning")
            param_grid = self.param_grid[model_name]
            grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            logging.info("Hyperparameter tuning completed")

            # save the model
            logging.info("Saving model")
            model_path = os.path.join("artifacts/models", f"{model_name}.pkl")
            self.utils.save_model(grid_search.best_estimator_, model_path)
            logging.info(f"Model saved at {model_path}")

            return grid_search.best_estimator_

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            self.exception_handler(e, sys)


if __name__ == "__main__":
    config = ModelTrainerConfig(model_path="artifacts/models")
    model_trainer = ModelTrainer(config)
    score = model_trainer.train_model("Lasso", "artifacts/train_data.csv")
    print(" ================ ")
    print(score)

# Path: src/components/model_trainer.py
