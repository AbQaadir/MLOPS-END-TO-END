from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


def main():
    data_ingestion_config = DataIngestionConfig()
    data_ingestion = DataIngestion(data_ingestion_config)
    train_data, test_data = data_ingestion.execute()

    print(train_data.head())
    print(test_data.head())

    data_transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation(data_transformation_config)
    X_train, y_train = data_transformation.transform_data(train_data)
    print(X_train.head())
    print(y_train.head())

    model_trainer_config = ModelTrainerConfig()
    model_trainer = ModelTrainer(model_trainer_config)
    gride_search = model_trainer.train_model("Lasso", train_data)
    print(gride_search)


if __name__ == "__main__":
    main()


# path: src/pipeline/training_pipeline.py
