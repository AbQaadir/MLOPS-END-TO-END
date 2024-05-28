# Path src/logger/my_logging.py

import logging
import os
from datetime import datetime

# Set the log file name format
LOG_FILE_FORMAT = "%Y-%m-%d-%H-%M-%S.log"

# Set the log path
LOG_PATH = "logs"

# Create the log path if it doesn't exist
log_path = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(log_path, exist_ok=True)


def get_log_file_name():
    """Get the log file name"""
    return f"{datetime.now().strftime(LOG_FILE_FORMAT)}"


def setup_logging():
    """Setup the logging configuration"""
    log_file_name = get_log_file_name()
    log_file_path = os.path.join(log_path, log_file_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )


# Setup the logging configuration
setup_logging()