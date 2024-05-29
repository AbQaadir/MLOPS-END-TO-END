from setuptools import setup, find_packages
from typing import List

setup(
    author='Qaadir',
    name='ProductionReadyMLOPS',
    author_email="qaadir.inbox@gmail.com",
    version='0.0.1',
    install_requires=["scikit-learn","pandas","numpy"],
    packages=find_packages()
)
