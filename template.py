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

package_name = "mongodb_connect"

list_of_files = [
    ".github/workflows/ci.yaml",
    "src/__init__.py",
    f"src/{package_name}/__init__.py",
    f"src/{package_name}/mongo_crud.py",
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "experiments/experiments.ipynb",
]

# Call the function to create files
create_files(list_of_files)