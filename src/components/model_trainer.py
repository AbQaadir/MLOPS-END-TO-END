from src.exception.my_exception import CustomExceptionHandler
from src.logger.my_logging import get_logger
from src.utils.utils import save_object, load_object, evaluate_model


class ModelTrainerConfig:
    def __init__(self, epochs: int, batch_size: int, learning_rate: float):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def __str__(self):
        return f"ModelTrainerConfig(epochs={self.epochs}, batch_size={self.batch_size}, learning_rate={self.learning_rate})"

    def __repr__(self):
        return f"ModelTrainerConfig(epochs={self.epochs}, batch_size={self.batch_size}, learning_rate={self.learning_rate})"


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_model(self, model, X_train, y_train, X_test, y_test):
        try:
            logger = get_logger(__name__)
            logger.info("Model training started")
            model.fit(
                X_train,
                y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_data=(X_test, y_test),
            )
            logger.info("Model training completed")
            return model
        except Exception as e:
            exception_handler = CustomExceptionHandler(e, sys)
            logger.error(exception_handler)
            raise e
