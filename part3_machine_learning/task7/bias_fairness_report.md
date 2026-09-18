# Bias, Fairness, Governance and Sustainability Report

## 1. Purpose

This report reviews the responsible-AI considerations associated with the Smart City Traffic Intelligence project.

The project uses the Metro Interstate Traffic Volume dataset to analyse traffic patterns and develop classification, regression, clustering, recommendation and deployment simulations.

The analysis is intended as a learning and demonstration project. It should not be treated as a production traffic-management or accident-prediction system.

---

## 2. Data Coverage and Sampling Limitations

The dataset represents traffic observations from a single traffic corridor.

This creates an important coverage limitation.

Patterns learned from this dataset may reflect:

* the characteristics of this particular road corridor
* local commuting behaviour
* local weather patterns
* local holiday patterns
* the historical period represented by the dataset

The models should therefore not be assumed to generalise directly to:

* other roads
* other cities
* other countries
* different transport networks
* substantially different time periods

A wider production system would require data from multiple locations and more recent observations before broader conclusions could be made.

---

## 3. Proxy Accident-Risk Label

No real accident dataset was supplied for the capstone.

Therefore, a proxy `high_risk` label was created as required by the assignment.

The proxy is defined using:

* High or Severe congestion
* together with severe or low-visibility weather

The resulting target distribution was:

* `high_risk = 1` -> 5,438 records
* `low_risk = 0` -> 42,749 records

This proxy must not be interpreted as actual accident likelihood.

It represents a combination of traffic congestion and adverse weather conditions rather than observed accidents.

For example:

```text
High congestion + adverse weather
-> high_risk = 1
```

does not mean that an accident actually occurred.

The classification results therefore demonstrate the machine-learning workflow rather than the ability to predict real accidents.

---

## 4. Risk of Target Leakage

The proxy `high_risk` label is partly derived from congestion, while congestion itself is calculated from `traffic_volume`.

For this reason, the following variables were excluded from the classification-model inputs:

* `traffic_volume`
* `traffic_volume_scaled`
* congestion category

Including these variables would indirectly provide the model with information used to create the target.

This would create target leakage and could produce unrealistically strong evaluation results.

---

## 5. Class Imbalance

The proxy classification target is imbalanced:

```text
high_risk = 1 -> 5,438 records  ≈ 11.3%
low_risk  = 0 -> 42,749 records ≈ 88.7%
```

Because of this imbalance, classification performance was not evaluated using accuracy alone.

The project also reports:

* Precision
* Recall
* F1-score
* ROC AUC

This provides a more complete view of how effectively the models identify the minority `high_risk = 1` class.

---

## 6. Uneven Model Errors

Prediction errors may not be distributed equally across all traffic conditions.

Potential differences could occur across:

* peak and non-peak hours
* weekdays and weekends
* normal and severe weather
* low-visibility conditions
* holidays and non-holidays
* low and high congestion periods

For example, rare weather conditions such as Squall, Smoke or severe Snow may contain substantially fewer observations than common conditions such as Clouds or Clear weather.

As a result, model performance for rare conditions may be less reliable.

A production system should evaluate model errors separately across these operational groups rather than relying only on overall performance metrics.

---

## 7. Recommendation-System Limitations

The recommendation system uses historical traffic patterns to recommend lower-traffic travel periods.

Because the dataset represents a single corridor, the system recommends **travel timing rather than alternative routes**.

Recommendations are restricted to practical travel hours from 06:00 to 22:00.

This avoids recommending overnight periods purely because they have the lowest historical traffic.

However, the recommendation system does not currently consider:

* live traffic incidents
* road closures
* construction
* public events
* real-time weather forecasts
* individual traveller requirements

The recommendations should therefore be interpreted as historical guidance rather than real-time navigation advice.

---

## 8. Model Explainability

SHAP was used to understand the main factors influencing traffic-volume predictions.

The explainability analysis showed that time-of-day variables had the strongest influence, followed by day-of-week and weekend effects.

Weather variables generally had smaller effects.

Explainability is important because traffic-management decisions should not rely solely on model predictions without understanding the factors that influence those predictions.

Permutation SHAP was applied to a comparable Random Forest Regressor trained on the same traffic-volume prediction problem as the neural network.

The SHAP analysis used a representative sample of 200 test records to control computational cost.

---

## 9. Human Oversight and Governance

A real traffic-management system should not automatically make major operational decisions solely from model predictions.

Appropriate governance should include:

* human review of important decisions
* documented model versions
* experiment tracking
* performance monitoring
* drift detection
* clear alert thresholds
* periodic retraining
* validation against current traffic conditions
* controlled deployment and rollback procedures

This project demonstrates several of these practices through:

* MLflow experiment tracking
* model version documentation
* FastAPI deployment simulation
* prediction-error monitoring
* feature-drift monitoring
* PASS / ALERT status reporting

Model predictions should support human decision-making rather than replace responsible operational oversight.

---

## 10. Monitoring and Model Drift

Traffic behaviour can change over time because of factors such as:

* new roads
* population changes
* changes in commuting behaviour
* public transport development
* economic activity
* weather patterns
* policy changes

The project therefore includes simulated monitoring for:

* prediction-error drift
* feature-distribution drift

The current monitoring simulation returned:

`PASS / Normal`

However, this monitoring uses historical data rather than live production observations.

A real deployment would require continuously collected current data and clearly defined retraining procedures.

---

## 11. Sustainability

Different modelling approaches have different computational costs.

The project uses:

* Linear Regression
* Logistic Regression
* Random Forest models
* K-means
* association-rule mining
* a PyTorch neural network
* SHAP explainability

The Random Forest Regressor achieved slightly better predictive performance than the neural network:

```text
Random Forest:
MAE       -> 269.83
R-squared -> 0.9449

Neural Network:
MAE       -> 280.12
R-squared -> 0.9437
```

Because the Random Forest achieved slightly stronger performance without requiring repeated GPU-based neural-network training, it may be a more resource-efficient choice for this particular dataset.

The project nevertheless includes the neural network to demonstrate deep-learning capability.

For a real deployment, model complexity should be justified by measurable improvements in performance or operational value.

---

## 12. Responsible Use

The models developed in this project should be considered analytical prototypes.

They should not be used directly for:

* real accident prediction
* emergency response
* safety-critical traffic control
* enforcement decisions
* decisions affecting individual road users

without additional validated data, external testing, governance and human oversight.

The proxy accident-risk label in particular must not be presented as evidence of actual accident probability.

---

## 13. Conclusion

The project demonstrates that machine learning can identify useful traffic patterns and predict traffic volume effectively.

However, responsible deployment requires more than predictive accuracy.

Important considerations include:

* data representativeness
* proxy-label limitations
* class imbalance
* unequal error distribution
* model explainability
* human oversight
* monitoring and drift detection
* computational sustainability

These limitations and safeguards should be considered before any model developed in this project is applied to real-world traffic-management decisions.
