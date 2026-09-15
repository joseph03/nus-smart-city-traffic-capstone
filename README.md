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
- `part1_data_analytics/` - SQL, SQL db, python codes
- `part2_python/` - Python data pipeline, feature engineering, visualisation and mini application
- `part3_machine_learning/` - machine learning, deep learning, explainability and MLOps

## Task 1.1 - Load and Clean Dataset
- `data/`
  - raw/Metro_Interstate_Traffic_Volume.csv  - raw dataset
  - processed/traffic_clean.csv              - clean dataset
- `part1_data_analytics/` 
  - traffic.db - SQLite database containing raw and cleaned traffic tables
    - traffic - table with raw traffic records
    - traffic_clean - table with cleaned traffic records
  - load_sqlite.py - load raw dataset to traffic table
  - inspect_data.py - inspect raw dataset
  - clean_data.py - create traffic_clean.csv, traffic_clean table 

- cleaning done
  - Preserved the original raw CSV and raw SQLite traffic table unchanged.
  - Removed 17 exact duplicate rows.
  - Replaced missing holiday values with None.
  - Standardised weather_description text to lowercase.
  - Replaced 10 invalid temperature readings of 0 K using the median temperature for the corresponding month.
  - Replaced one extreme rainfall value of 9831.3 mm with the median positive rainfall for the same month.
  - Verified that the cleaned dataset contains no duplicate rows, invalid dates, temperatures at or below 0 K, extreme rainfall values, invalid cloud-cover values, or negative traffic volumes.
  - Saved the cleaned data as data/processed/traffic_clean.csv and as the traffic_clean table in traffic.db.

## Task 1.2 - Annual Traffic Trend Findings
- `part1_data_analytics/` 
  - task1_2_annual_traffic.sql
  - task1_2a_check_days.sql

* Annual totals for 2012, 2014 and 2015 should be interpreted cautiously because those years have incomplete data coverage.
* 2012 contains only 91 days of records, 2014 contains 214 days, and 2015 contains 195 days.
* Therefore, large year-on-year percentage changes involving these years mainly reflect differences in data availability rather than true traffic growth or decline.
* 2016 and 2017 both have complete yearly coverage.
* Traffic volume increased from 29,471,608 in 2016 to 35,393,801 in 2017, an increase of approximately 20.09%.

## Task 1.3 - Holiday Temperature Findings
- `part1_data_analytics/` 
  - task1_3_holiday_temperature.sql
  - task1_3a_duplicate_holidays.sql
  - task1_3b_enhanced_holiday_temperature.sql

* Labor Day temperatures were fairly stable across the three available years: 21.87°C in 2015, 20.02°C in 2016, and 22.39°C in 2017. 
* Labor Day traffic volume also remained relatively similar, ranging from 973 to 1,064 vehicles. This suggests no obvious strong relationship between the small temperature differences and traffic volume for these observations. 
* New Years Day was much colder than Labor Day. The recorded temperature increased from -7.21°C in 2016 to -2.53°C in 2017. 
* Traffic volume on New Years Day decreased from 1,513 vehicles in 2016 to 798 vehicles in 2017, even though 2017 was warmer. This indicates that temperature alone does not explain the traffic difference. 
* No New Years Day record is available for 2015 because the 2015 dataset begins only in June. 
* These results should be interpreted cautiously because the holiday field represents specific recorded timestamps, not a complete set of hourly observations for the whole holiday. 
* There is no clear evidence here that temperature alone drove holiday traffic volume.

