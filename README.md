# NUS Smart City Traffic Capstone

Capstone Project: Smart City Traffic Intelligence — From Data Analytics to AI-Powered Mobility.

## Hardware and GPU Notes

Most of this project can run on CPU, including:

- SQL/statistical analysis
- Python data cleaning and feature engineering
- Matplotlib visualisations
- Scikit-learn models

A GPU is optional but recommended for the deep-learning component.

This project was developed in WSL2 with an NVIDIA GPU-enabled TensorFlow environment.
Users without a compatible GPU can still run the deep-learning code on CPU, although training may be slower.

## Conda environments

The environment specification is stored in:
- environment.yml
- environment-tensorflow.yml

To recreate the environment:

conda env create -f environment.yml

conda activate pytorch310

conda env create -f environment-tensorflow.yml

conda activate tf_wsl2_310

## Project Structure
- `data/` - raw and processed traffic data
- `part1_data_analytics/` - SQL, statistics, probability and Power BI
  - traffic.db - SQLite database containing raw and cleaned traffic tables
    - traffic - table with raw traffic records
- `part2_python/` - Python data pipeline, feature engineering, visualisation and mini application
- `part3_machine_learning/` - machine learning, deep learning, explainability and MLOps
