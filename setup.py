from setuptools import setup, find_packages

setup(
    name="trading-engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "xgboost",
    ],
    description="Real-Time Trading Engine",
)
