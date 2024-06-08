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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


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
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "ElasticNet": ElasticNet(),
            "RandomForest": RandomForestRegressor(),
            "XGBoost": XGBRegressor(),
        }
        self.param_grid = {
            "LinearRegression": {
                "normalize": [True, False],
                "n_jobs": [-1, 1, 2],
            },
            "Lasso": {
                "alpha": [0.1, 1.0, 10.0],
            },
            "Ridge": {
                "alpha": [0.1, 1.0, 10.0],
            },
            "ElasticNet": {
                "elastic_net_alpha": [0.1, 1.0, 10.0],
                "elastic_net_l1_ratio": [0.1, 0.5, 0.9],
            },
            "RandomForest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10],
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.1, 0.5, 1.0],
                "gamma": [0.0, 0.1, 0.5],
                "subsample": [0.5, 0.8, 1.0],
            },
        }

    # save the model in the model_path directory
    def save_model(self, model, model_path):
        try:
            with open(model_path, "wb") as file:
                pickle.dump(model, file)
        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")
            self.exception_handler(e, sys)

    # train all the model and save the best model
    def train_model(
        self,
        train_arr: np.ndarray,
        test_arr: np.ndarray,
    ):
        # Split the data into features and target
        X_train = train_arr[:, :-1]
        y_train = train_arr[:, -1]
        X_test = test_arr[:, :-1]
        y_test = test_arr[:, -1]

        # Train all the models and save the best model
        best_model = None
        best_score = -np.inf
        best_model_name = ""

        for model_name, model in self.models.items():
            try:
                logging.info(f"Training model: {model_name}")

                grid_search = GridSearchCV(
                    model, self.param_grid[model_name], cv=5, n_jobs=-1
                )
                grid_search.fit(X_train, y_train)

                y_pred = grid_search.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                logging.info(
                    f"Model: {model_name}, R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}"
                )

                if r2 > best_score:
                    best_score = r2
                    best_model = grid_search.best_estimator_
                    best_model_name = model_name

            except Exception as e:
                logging.error(f"Failed to train model {model_name}: %s", str(e))
                continue

        if best_model:
            try:
                save_path = os.path.join(
                    self.config.model_path, f"{best_model_name}.pkl"
                )
                self.save_model(best_model, save_path)
                logging.info(
                    f"Best model {best_model_name} saved successfully at {save_path}"
                )
            except Exception as e:
                logging.error(f"Failed to save the best model: %s", str(e))
        else:
            logging.error("No model was successfully trained")

        logging.info("Model training completed successfully")


# if __name__ == "__main__":
#     config = ModelTrainerConfig()
#     model_trainer = ModelTrainer(config)
#     train_arr = np.random.rand(100, 10)
#     test_arr = np.random.rand(100, 10)
#     model_trainer.train_model(train_arr, test_arr)


# path: src/components/model_trainer.py
