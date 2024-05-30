import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.logger.logging import logging
from src.exception.exception import CustomExceptionHandler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class Utils:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exception_handler = CustomExceptionHandler

    def load_data(self, data_path):
        try:
            data = pd.read_csv(data_path, low_memory=False)
            self.logger.info(f"Data loaded from {data_path}")
            return data
        except Exception as e:
            self.logger.error(self.exception_handler(e, sys))
            raise

    def data_split(self, data: pd.DataFrame):
        X = data.drop(columns=["id", "price"])
        y = data["price"]
        return X, y

    def train_test_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification_report = classification_report(y_test, y_pred)
        confusion_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, classification_report, confusion_matrix

    def save_model(self, model, model_path):
        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        self.logger.info(f"Model saved at {model_path}")

    def load_model(self, model_path):
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        self.logger.info(f"Model loaded from {model_path}")
        return model

    def save_preprocessor(self, preprocessor, preprocessor_path):
        with open(preprocessor_path, "wb") as file:
            pickle.dump(preprocessor, file)
        self.logger.info(f"Preprocessor saved at {preprocessor_path}")

    def load_preprocessor(self, preprocessor_path):
        with open(preprocessor_path, "rb") as file:
            preprocessor = pickle.load(file)
        self.logger.info(f"Preprocessor loaded from {preprocessor_path}")
        return preprocessor

    def make_pipeline(self, preprocessor, model):
        return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
