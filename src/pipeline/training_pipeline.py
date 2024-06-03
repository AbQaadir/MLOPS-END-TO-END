from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


def main():
    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)
    train_data_path, test_data_path = data_ingestion.execute()

    config = DataTransformationConfig()
    data_tansformation = DataTransformation(config)
    train_arr, test_arr = data_tansformation.transform_data(train_data_path, test_data_path)
    
    config = ModelTrainerConfig()
    model_trainer = ModelTrainer(config)
    model_trainer.train_model(train_arr, test_arr)


if __name__ == "__main__":
    main()


# path: src/pipeline/training_pipeline.py
