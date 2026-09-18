# Model Version History

## Classification Models

### Version C1 - Logistic Regression

Purpose:
Predict the proxy `high_risk` classification target.

Performance:

- Accuracy: 0.9788
- Precision: 0.9123
- Recall: 0.8989
- F1-score: 0.9056
- ROC AUC: 0.9938

Status:
Baseline classification model.

---

### Version C2 - Random Forest Classifier

Purpose:
Predict the proxy `high_risk` classification target.

Performance:

- Accuracy: 0.9841
- Precision: 0.9277
- Recall: 0.9320
- F1-score: 0.9298
- ROC AUC: 0.9970

Status:
Stronger classification model based on the evaluation metrics.

---

## Regression Models

### Version R1 - Linear Regression

Purpose:
Predict `traffic_volume`.

Performance:

- MAE: 834.09 vehicles
- R-squared: 0.7092

Status:
Baseline regression model.

---

### Version R2 - Random Forest Regressor

Purpose:
Predict `traffic_volume`.

Performance:

- MAE: 269.83 vehicles
- R-squared: 0.9449

Status:
Stronger regression model based on the evaluation metrics.

---

### Version R3 - PyTorch Neural Network

Purpose:
Predict `traffic_volume` using deep learning.

Architecture:

- 21 input features
- 64-neuron hidden layer
- ReLU activation
- 32-neuron hidden layer
- ReLU activation
- single regression output

Performance:

- MAE: 280.12 vehicles
- R-squared: 0.9437

Status:
Deep-learning model with performance close to the Random Forest Regressor.