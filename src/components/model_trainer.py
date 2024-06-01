import pandas as pd
import numpy as np
import pickle
import os
import sys

from src.exception.exception import (
    CustomExceptionHandler,
)  # Custom exception handler module
from src.logger.logging import logging  # Custom logging module
from dataclasses import dataclass

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.utils.utils import Utils


@dataclass
class ModelTrainerConfig:
    model_path: str = "artifacts/models"


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.exception_handler = CustomExceptionHandler
        self.logger = logging.getLogger(__name__)
        self.models = {
            "LinearRegression": LinearRegression(),
            "Lasso": Lasso(max_iter=100000),
            "Ridge": Ridge(max_iter=100000),
            "Elasticnet": ElasticNet(max_iter=100000),
            "Randomforest": RandomForestRegressor(),
            "xgboost": XGBRegressor(),
        }
        self.param_grid = {
            "Lasso": {"regressor__alpha": [0.1, 1.0, 10.0]},
            "Ridge": {"regressor__alpha": [0.1, 1.0, 10.0]},
            "ElasticNet": {
                "regressor__alpha": [0.1, 1.0, 10.0],
                "regressor__l1_ratio": [0.1, 0.5, 0.9],
            },
            "RandomForest": {"n_estimators": [50, 100, 200]},
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
            },
        }

    def load_preprocessor(self):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            with open(preprocessor_path, "rb") as file:
                preprocessor = pickle.load(file)
            return preprocessor
        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")
            self.exception_handler(e, sys)

    # save the model in the model_path directory
    def save_model(self, model, model_path):
        try:
            with open(model_path, "wb") as file:
                pickle.dump(model, file)
        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")
            self.exception_handler(e, sys)

    # train the model using the train_data
    def train_model(self, model_name, train_data):
        try:
            # Split the data into X and y
            logging.info("Splitting the data into X and y")
            X_train = train_data.drop("price", axis=1)
            y_train = train_data["price"]
            logging.info("Data split successfully")

            # Load the model
            self.logger.info(f"Training {model_name} model")
            model = self.models[model_name]
            self.logger.info(f"Model {model_name} loaded successfully")

            # Load the preprocessor
            self.logger.info("Loading preprocessor")
            preprocessor = self.load_preprocessor()
            self.logger.info("Preprocessor loaded successfully")

            # Create a pipeline
            self.logger.info("Creating pipeline")
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", model),
                ]
            )
            self.logger.info("Pipeline created successfully")

            # Perform grid search
            self.logger.info("Performing grid search")
            grid_search = GridSearchCV(
                pipeline, self.param_grid[model_name], cv=5, n_jobs=-1
            )
            self.logger.info("Grid search completed successfully")

            # Fit the model
            self.logger.info("Fitting the model")
            grid_search.fit(X_train, y_train)
            self.logger.info("Model fitted successfully")

            # Save the model
            self.logger.info("Saving the model")
            model_path = os.path.join(self.config.model_path, f"{model_name}.pkl")
            self.save_model(grid_search, model_path)
            self.logger.info("Model saved successfully")

            return grid_search

        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")
            self.exception_handler(e, sys)


# if __name__ == "__main__":
# model_trainer_config = ModelTrainerConfig()
# model_trainer = ModelTrainer(model_trainer_config)
# train_data = pd.read_csv("data/train_data.csv")
# model_trainer.train_model("Lasso", train_data)

# path: src/components/model_trainer.py
