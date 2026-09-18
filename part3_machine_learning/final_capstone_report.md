# Smart City Traffic Intelligence
## Part 3 Final Machine Learning, Deep Learning and AI Report

## 1. Introduction

Part 3 of the Smart City Traffic Intelligence capstone extends the data-cleaning, feature-engineering and exploratory work completed in Parts 1 and 2 into machine learning, deep learning, model explainability, recommendation, deployment simulation and responsible AI.

The main input dataset is:

`data/processed/traffic_features_part2.csv`

The dataset contains 48,187 cleaned traffic observations and 30 columns.

The objectives of Part 3 are to:

- develop supervised machine-learning models for classification and regression
- apply unsupervised-learning techniques to discover traffic patterns
- implement a deep-learning model
- explain model behaviour using SHAP
- demonstrate experiment tracking using MLflow
- develop a practical traffic recommendation system
- simulate model deployment, monitoring and alerting
- evaluate bias, fairness, governance and sustainability considerations

Because the provided dataset does not contain real accident records, the classification task uses a proxy accident-risk label. This proxy is used only to demonstrate the machine-learning workflow and must not be interpreted as actual accident probability.

## 2. Data and Feature Preparation

The feature-engineered dataset produced in Part 2 contains time, weather and traffic-related variables.

A common set of 21 engineered input features was used for the supervised-learning models. These include:

- `hour_sin`
- `hour_cos`
- `day_sin`
- `day_cos`
- weekend indicator
- holiday flag
- scaled temperature
- rainfall
- snowfall
- cloud coverage
- one-hot encoded weather categories

Cyclical encodings were used for hour and day so that recurring time patterns could be represented more appropriately than with simple linear numeric values.

For classification, variables directly connected to the proxy target were excluded to prevent target leakage:

- `traffic_volume`
- `traffic_volume_scaled`
- congestion category

For regression, the prediction target is `traffic_volume`, so `traffic_volume_scaled` was also excluded from the regression input features.

## 3. Supervised Machine Learning

### 3.1 Classification Problem

No real accident dataset was available. A proxy binary target named `high_risk` was therefore created.

Traffic volume was divided into four congestion categories using quartiles:

- Low: traffic volume <= Q1 = 1,192.50
- Medium: traffic volume <= Q2 = 3,379.00
- High: traffic volume <= Q3 = 4,933.00
- Severe: traffic volume > Q3

The proxy `high_risk` condition was defined as:

- High or Severe congestion
- AND severe weather or low-visibility weather

The resulting target distribution was:

```text
high_risk = 1 -> 5,438 records
low_risk  = 0 -> 42,749 records
```

The target is imbalanced, with the high-risk class representing approximately 11.3% of observations.

An 80% training and 20% testing split was used with stratification so that the class distribution remained consistent across the training and test datasets.

### 3.2 Logistic Regression

The Logistic Regression model achieved:

- Accuracy: 0.9788
- Precision: 0.9123
- Recall: 0.8989
- F1-score: 0.9056
- ROC AUC: 0.9938

These results show that a relatively simple linear classification model can identify the proxy high-risk condition effectively.

### 3.3 Random Forest Classification

The Random Forest Classifier achieved:

- Accuracy: 0.9841
- Precision: 0.9277
- Recall: 0.9320
- F1-score: 0.9298
- ROC AUC: 0.9970

The Random Forest Classifier achieved stronger results than Logistic Regression across all reported evaluation metrics.

However, the high performance must be interpreted carefully because the target is a constructed proxy rather than a real accident label.

## 4. Traffic-Volume Regression

The regression task predicts `traffic_volume`.

Two machine-learning algorithms were evaluated:

- Linear Regression
- Random Forest Regressor

### 4.1 Linear Regression

The Linear Regression model achieved:

- MAE: 834.09 vehicles
- R-squared: 0.7092

The model explains approximately 70.9% of the observed variation in traffic volume.

This provides a useful baseline but indicates that traffic behaviour contains nonlinear relationships that a basic linear model does not fully capture.

### 4.2 Random Forest Regressor

The Random Forest Regressor achieved:

- MAE: 269.83 vehicles
- R-squared: 0.9449

The Random Forest substantially improved prediction accuracy compared with Linear Regression.

An R-squared of approximately 0.945 indicates that the model explains a large proportion of the variation in traffic volume within the test data.

The Random Forest Regressor was therefore selected for the later deployment simulation.

## 5. Unsupervised Machine Learning

Unsupervised learning was used to discover patterns without a predefined prediction target.

Two techniques were implemented:

- K-means clustering
- association-rule mining

## 6. K-means Traffic Clustering

K-means clustering used three variables:

- hour
- traffic volume
- weather severity

Weather severity was encoded as:

- 0 = relatively normal weather
- 1 = rain or drizzle
- 2 = low-visibility weather
- 3 = severe weather

The variables were standardized before applying K-means.

Four clusters were created.

### Cluster 0

- Records: 8,309
- Average hour: 12.42
- Average traffic volume: 4,072.76
- Average weather severity: 2.38

Interpretation: daytime traffic under relatively poor or severe weather conditions.

### Cluster 1

- Records: 13,209
- Average hour: 2.95
- Average traffic volume: 833.91
- Average weather severity: 0.73

Interpretation: overnight conditions with relatively low traffic volume.

### Cluster 2

- Records: 17,449
- Average hour: 12.35
- Average traffic volume: 5,075.93
- Average weather severity: 0.20

Interpretation: busy daytime traffic under generally normal weather conditions.

### Cluster 3

- Records: 9,220
- Average hour: 20.77
- Average traffic volume: 2,564.61
- Average weather severity: 0.31

Interpretation: evening traffic with moderate traffic volume and generally normal weather.

The clustering results demonstrate that traffic conditions can be separated into meaningful combinations of time, traffic demand and weather severity.

## 7. Association-Rule Mining

Association-rule mining was applied to categorical representations of:

- time of day
- weekday or weekend
- weather group
- congestion level

The transaction matrix contained:

- 48,187 observations
- 14 categorical items

Apriori analysis generated:

- 149 frequent itemsets
- 62 congestion-prediction rules after filtering

### Representative Rule 1: Weekend Night -> Low Congestion

- Support: 0.0644
- Confidence: 0.8861
- Lift: 3.5443

Approximately 88.6% of records meeting the Weekend Night condition were associated with Low congestion.

The lift of 3.54 indicates that Low congestion was substantially more common under these conditions than in the dataset overall.

### Representative Rule 2: Weekend Afternoon + Normal Weather -> High Congestion

- Support: 0.0387
- Confidence: 0.8454
- Lift: 3.3797

This indicates that High congestion was strongly associated with weekend afternoons under normal weather in the historical dataset.

Overall, the association rules show that time of day and day type have important relationships with congestion.

## 8. Deep Learning

A feed-forward neural network was developed using PyTorch.

The model used the same 21 engineered input features as the regression models.

The architecture consisted of:

- input layer
- 64-neuron hidden layer with ReLU
- 32-neuron hidden layer with ReLU
- one regression output

The model was trained for:

- 50 epochs
- Adam optimizer
- mean squared error loss

CUDA was successfully used for model training.

### Neural Network Results

The neural network achieved:

- MAE: 280.12 vehicles
- R-squared: 0.9437

The Random Forest Regressor remained slightly stronger:

```text
Random Forest:
MAE       -> 269.83
R-squared -> 0.9449

Neural Network:
MAE       -> 280.12
R-squared -> 0.9437
```

This result demonstrates that increased model complexity does not automatically produce superior performance.

## 9. Model Explainability Using SHAP

SHAP was used to understand the factors influencing traffic-volume predictions.

A model-agnostic Permutation SHAP explainer was applied to a comparable Random Forest Regressor trained on the same traffic-volume prediction problem.

Permutation SHAP was used because the tree-specific SHAP implementation produced a low-level segmentation fault in the project environment.

A representative sample of 200 test observations was used to control computational cost.

The most influential variables identified by SHAP were:

- `hour_cos`
- `hour_sin`
- `day_sin`
- `weekend`
- `temp_scaled`
- `day_cos`

The results show that time of day is the dominant contributor to traffic-volume predictions.

Because `hour_sin` and `hour_cos` jointly encode hour of day, they should be interpreted together rather than as independent variables.

Weekend observations generally reduced predicted traffic volume, which is consistent with the earlier descriptive analysis showing lower weekend traffic.

Weather variables generally contributed less to overall predictions than recurring time-based patterns.

## 10. Advanced AI and MLflow Experiment Tracking

MLflow was selected as the advanced AI technique because it supports structured experiment tracking and integrates naturally with the MLOps workflow.

The MLflow experiment is named `smart_city_traffic_models`.

The experiment tracks four Task 1 models:

- Logistic Regression Classification
- Random Forest Classification
- Linear Regression
- Random Forest Regression

For classification models, MLflow records:

- model type
- model parameters
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC

For regression models, MLflow records:

- model type
- model parameters
- MAE
- R-squared

The project uses a local SQLite MLflow backend:

`part3_machine_learning/mlflow.db`

The normal program execution log remains separate in:

`part3_machine_learning/part3.log`

MLflow improves reproducibility by preserving a structured history of model experiments and evaluation results.

## 11. Traffic Recommendation System

Because the dataset represents only one traffic corridor, meaningful route-selection recommendations cannot be produced.

The recommendation system therefore focuses on travel timing rather than route choice.

The system:

- filters historical records by weekday or weekend
- optionally filters by weather condition
- restricts recommendations to practical travel hours between 06:00 and 22:00
- calculates average traffic volume by hour
- identifies the historical hour with the lowest traffic volume
- returns a plain-language recommendation

For a weekday journey, the historical analysis recommends **21:00 to 22:00**, with an average traffic volume of approximately **2,673 vehicles**.

The recommendation remains historical rather than real-time because the dataset does not contain live incident, road-closure, construction or current weather information.

## 12. MLOps and Deployment Simulation

The MLOps component demonstrates how a trained model could move from development into operational use.

The workflow includes:

- model versioning
- MLflow experiment tracking
- model deployment simulation
- prediction-error monitoring
- feature-drift monitoring
- operational alerting

## 13. Model Versioning

The project documents the main models using version identifiers.

Classification models:

- C1 - Logistic Regression
- C2 - Random Forest Classifier

Regression models:

- R1 - Linear Regression
- R2 - Random Forest Regressor
- R3 - PyTorch Neural Network

The Random Forest Regressor was selected for deployment because it achieved the strongest regression results.

## 14. FastAPI Deployment Simulation

The deployment simulation was implemented using FastAPI.

The deployed Random Forest model artifact is:

`part3_machine_learning/models/random_forest_regressor.joblib`

The artifact contains:

- the trained Random Forest Regressor
- feature-column definitions
- temperature mean
- temperature standard deviation

The API provides two endpoints.

### GET /

Purpose: check whether the service is running.

Successful response:

```json
{
  "status": "PASS",
  "message": "Smart City Traffic Prediction API is running."
}
```

### POST /predict

The endpoint accepts:

- hour
- day of week
- holiday indicator
- temperature
- rainfall
- snowfall
- cloud coverage
- main weather condition

A successful example prediction returned:

```json
{
  "status": "PASS",
  "predicted_traffic_volume": 5555.76
}
```

The Random Forest artifact is approximately 623 MB and exceeds GitHub's standard 100 MB file-size limit.

It is therefore excluded from version control using `.gitignore`.

The model artifact can be reproduced locally by running:

`python part3_machine_learning/supervised_models.py`

## 15. Model Monitoring

The project simulates two forms of operational monitoring:

- prediction-error drift
- feature-distribution drift

### 15.1 Prediction-Error Monitoring

The baseline Random Forest test MAE was **269.83**.

The simulated recent-window MAE was **133.24**.

The corresponding MAE change was **-50.62%**.

The alert rule is:

- ALERT if recent MAE increases by more than 25% relative to baseline

Because the simulated recent-window error was lower than the baseline, no prediction-error alert was triggered.

However, this monitoring uses historical data and should not be interpreted as evidence that future production performance will improve.

### 15.2 Feature-Drift Monitoring

Feature drift is estimated using standardized mean differences.

The alert threshold is **0.25**.

Observed scores included:

- `hour_sin`: 0.0004
- `hour_cos`: 0.0150
- `day_sin`: 0.0093
- `day_cos`: 0.0072
- `weekend`: 0.0034
- `temp_scaled`: 0.0505
- `rain_1h`: 0.0282
- `snow_1h`: 0.0276
- `clouds_all`: 0.0562

All calculated values remained below the alert threshold.

The `holiday_flag` comparison was skipped because the reference sample had zero or unavailable standard deviation.

## 16. Operational Alerting

The monitoring system converts the results into one of two operational states:

- PASS / Normal
- ALERT / Requires investigation

The current simulated status is **PASS / Normal**.

The latest monitoring state is stored in:

`part3_machine_learning/mlops/monitoring_status.json`

This file is separate from MLflow.

## 17. Bias, Fairness and Responsible AI

The responsible-AI assessment identified several important limitations.

### 17.1 Data Coverage

The dataset represents observations from a single corridor.

The learned patterns may therefore reflect local traffic, commuting, weather and holiday patterns and should not automatically be generalized to other roads, cities or transport networks.

### 17.2 Proxy Accident-Risk Label

The `high_risk` target is not based on observed accidents.

It represents High or Severe congestion combined with adverse weather and must not be interpreted as actual accident likelihood.

### 17.3 Target Leakage

Variables associated with the proxy definition were excluded from classification inputs:

- `traffic_volume`
- `traffic_volume_scaled`
- congestion category

### 17.4 Class Imbalance

The proxy target contains approximately:

```text
high_risk = 1 -> 11.3%
low_risk  = 0 -> 88.7%
```

Classification was therefore evaluated using Precision, Recall, F1-score and ROC AUC in addition to Accuracy.

### 17.5 Fairness Limitations

The dataset does not contain demographic or protected attributes such as age, gender, ethnicity or socioeconomic status.

Demographic fairness metrics therefore cannot be meaningfully evaluated.

Fairness assessment in this project is limited to possible differences in model performance across observable traffic, time and weather conditions.

### 17.6 Uneven Model Errors

Model performance may differ across:

- peak and non-peak periods
- weekdays and weekends
- severe and normal weather
- low-visibility conditions
- holidays
- congestion levels

Rare weather conditions may contain fewer observations and therefore produce less reliable estimates.

## 18. Governance and Human Oversight

A real traffic-management system should not automatically make important operational decisions solely from model outputs.

Appropriate governance would include:

- human review
- model version documentation
- experiment tracking
- model-performance monitoring
- drift detection
- defined alert thresholds
- periodic validation
- retraining procedures
- controlled deployment
- rollback procedures

The project demonstrates several of these principles through MLflow, model versioning, FastAPI, monitoring and PASS / ALERT reporting.

## 19. Sustainability

Different machine-learning approaches require different computational resources.

The Random Forest Regressor achieved:

- MAE: 269.83
- R-squared: 0.9449

The PyTorch neural network achieved:

- MAE: 280.12
- R-squared: 0.9437

The Random Forest achieved slightly better performance without requiring repeated GPU-based neural-network training.

For this dataset, it may therefore provide a more resource-efficient solution.

## 20. Key Findings

The Part 3 analysis produced several important findings:

- recurring time patterns are the strongest predictors of traffic demand
- Random Forest models performed strongly for both classification and regression
- the neural network achieved strong performance but did not outperform the Random Forest Regressor
- K-means identified meaningful traffic-condition groups
- association-rule mining identified strong relationships between time/day combinations and congestion
- the recommendation system identified practical lower-traffic travel periods
- MLflow, FastAPI and monitoring demonstrated a basic end-to-end MLOps workflow

## 21. Limitations

The project has several important limitations.

The dataset:

- represents a single corridor
- contains historical rather than live traffic observations
- does not contain actual accident records
- does not contain demographic attributes
- does not represent a complete smart-city road network

The `high_risk` classification target is therefore only a proxy.

The recommendation system is historical rather than real-time.

The monitoring system also uses historical data as a simulation. The recent-data window should not be interpreted as an independent future-production test set.

A production implementation would require more recent and broader data, independent validation, continuous monitoring and stronger governance controls.

## 22. Conclusion

Part 3 demonstrates an end-to-end machine-learning workflow for smart-city traffic intelligence.

The project combines:

- supervised machine learning
- unsupervised learning
- deep learning
- explainable AI
- recommendation
- experiment tracking
- API deployment
- model monitoring
- responsible AI

The Random Forest Regressor produced the strongest traffic-volume prediction performance, while the neural network achieved comparable results.

SHAP analysis showed that recurring time patterns are the dominant drivers of predicted traffic demand.

K-means clustering and association-rule mining provided additional insight into recurring traffic conditions and congestion patterns.

MLflow improved experiment traceability, while FastAPI demonstrated how the selected model could be exposed through a prediction service.

The monitoring workflow demonstrated basic prediction-error and feature-drift checks together with operational PASS / ALERT reporting.

However, the project remains an analytical prototype. In particular, the proxy `high_risk` classification label must not be interpreted as actual accident probability.

A real smart-city deployment would require broader and more recent datasets, validated safety outcomes, independent production monitoring, stronger governance and continued human oversight.

Overall, the capstone demonstrates how data analytics, machine learning, deep learning, explainability and MLOps can be integrated into a coherent traffic-intelligence workflow while recognising the technical and responsible-AI limitations of the available data.
