import sys
import os
import numpy as np
import pickle
import pandas as pd
from src.exception.exception import CustomExceptionHandler
from src.logger.logging import logging
from dataclasses import dataclass

import pydantic
from pydantic import BaseModel


@dataclass
class PredictionPiplineConfig:
    model_path: str = "artifacts/models/XGBoost.pkl"
    test_data_path: str = "artifacts/test.csv"


class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity
    



class PredictionPipline:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.exception_handler = CustomExceptionHandler(self.logger)

    # load the model
    def load_model(self):
        try:
            with open(self.config.model_path, "rb") as file:
                model = pickle.load(file)
            return modele
        except Exception as e:
            self.exception_handler.handle_exception(e. sys)

    # load the preprocessor
    def load_preprocessor(self):
        try:
            with open("artifacts/preprocessor.pkl", "rb") as file:
                preprocessor = pickle.load(file)
            return preprocessor
        except Exception as e:
            self.exception_handler.handle_exception(e, sys)

    # predict the test data
    def predict(self, model, preprocessor, test_data):
        try:
            test_data = preprocessor.transform(test_data)
            prediction = model.predict(test_data)
            return prediction
        except Exception as e:
            self.exception_handler.handle_exception(e. sys)
            
    # execute the pipeline
    def execute(self):
        try:
            model = self.load_model()
            preprocessor = self.load_preprocessor()
            test_data = pd.read_csv(self.config.test_data_path)
            prediction = self.predict(model, preprocessor, test_data)
            return prediction
        except Exception as e:
            self.exception_handler.handle_exception(e,sys)


def main():
    config = PredictionPiplineConfig()
    pipeline = PredictionPipline(config)
    pipeline.execute()
    
    
if __name__ == "__main__":
    main()


# path : src/pipeline/prediction_pipeline.py