import os
from pathlib import Path


def create_files(file_paths):
    for filepath in file_paths:
        path = Path(filepath)

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create an empty file if it doesn't exist or if it is empty
        if not path.exists() or path.stat().st_size == 0:
            path.touch()


package_name = "MLOPS"

list_of_files = [
    "src/init.py",
    "src/components/init.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/pipeline/init.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    "src/utils/init.py",
    "src/utils/utils.py",
    "src/logger/logging.py",
    "src/exception/exception.py",
    "tests/unit/init.py",
    "tests/integration/init.py",
    "requirements.txt",
    "requirements_dev.txt",
    "experiments/experiments.ipynb",
]

# Call the function to create files
create_files(list_of_files)
