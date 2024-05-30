import pickle
from sklearn.model_selection import train_test_split
from src.logger.logging import get_logger
from src.exception.exception import CustomExceptionHandler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# create a save_object function to save the model is side the models directory
def save_object(obj, filename):
    try:
        # Save the model to the models directory
        with open(filename, "wb") as output:
            # Save the object to the file
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        exception_handler = CustomExceptionHandler(e, sys)
        logger = get_logger(__name__)
        logger.error(exception_handler)
        raise e


# create a load_object function to load the model from the models directory
def load_object(filename):
    try:
        # Load the model from the models directory
        with open(filename, "rb") as input:
            # Load the object from the file
            obj = pickle.load(input)
            return obj

    except Exception as e:
        exception_handler = CustomExceptionHandler(e, sys)
        logger = get_logger(__name__)
        logger.error(exception_handler)
        raise e


# create a function to evaluate the model
def evaluate_model(y_test, y_pred):
    try:
        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate the confusion matrix
        confusion = confusion_matrix(y_test, y_pred)

        # Generate the classification report
        report = classification_report(y_test, y_pred)

        return accuracy, confusion, report
    except Exception as e:
        exception_handler = CustomExceptionHandler(e, sys)
        logger = get_logger(__name__)
        logger.error(exception_handler)
        raise e
