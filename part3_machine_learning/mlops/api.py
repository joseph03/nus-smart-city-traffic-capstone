import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


logger = logging.getLogger(__name__)


# -------------------------------------------------
# FastAPI application
# -------------------------------------------------
app = FastAPI(
    title="Smart City Traffic Prediction API",
    description=(
        "Deployment simulation for predicting "
        "traffic volume using the trained "
        "Random Forest Regressor."
    ),
    version="1.0",
)


# -------------------------------------------------
# Input schema
# -------------------------------------------------
class TrafficInput(BaseModel):
    hour: int
    day_of_week: int
    holiday: bool
    temp: float
    rain_1h: float
    snow_1h: float
    clouds_all: float
    weather_main: str


# -------------------------------------------------
# Load saved model artifact
# -------------------------------------------------
MODEL_PATH = Path(
    "part3_machine_learning/models/"
    "random_forest_regressor.joblib"
)

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model artifact not found: {MODEL_PATH}"
    )

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
feature_columns = artifact["feature_columns"]
temp_mean = artifact["temp_mean"]
temp_std = artifact["temp_std"]

logger.info(
    "Random Forest deployment artifact loaded."
)


# -------------------------------------------------
# Create model input features
# -------------------------------------------------
def prepare_prediction_features(data):
    if not 0 <= data.hour <= 23:
        raise ValueError(
            "hour must be between 0 and 23."
        )

    if not 0 <= data.day_of_week <= 6:
        raise ValueError(
            "day_of_week must be between 0 and 6."
        )

    hour_sin = np.sin(
        2 * np.pi * data.hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * data.hour / 24
    )

    day_sin = np.sin(
        2 * np.pi * data.day_of_week / 7
    )

    day_cos = np.cos(
        2 * np.pi * data.day_of_week / 7
    )

    weekend = int(
        data.day_of_week >= 5
    )

    holiday_flag = int(
        data.holiday
    )

    temp_scaled = (
        data.temp - temp_mean
    ) / temp_std

    feature_values = {
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "weekend": weekend,
        "holiday_flag": holiday_flag,
        "temp_scaled": temp_scaled,
        "rain_1h": data.rain_1h,
        "snow_1h": data.snow_1h,
        "clouds_all": data.clouds_all,
    }

    # Set every one-hot weather feature to 0 first
    for column in feature_columns:
        if column.startswith("weather_"):
            feature_values[column] = 0

    weather_column = (
        f"weather_{data.weather_main}"
    )

    if weather_column not in feature_columns:
        raise ValueError(
            "Unsupported weather_main value: "
            f"{data.weather_main}"
        )

    feature_values[weather_column] = 1

    model_input = pd.DataFrame(
        [
            {
                column: feature_values[column]
                for column in feature_columns
            }
        ]
    )

    return model_input


# -------------------------------------------------
# Health endpoint
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "PASS",
        "message": (
            "Smart City Traffic Prediction API "
            "is running."
        ),
    }


# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
def predict_traffic(data: TrafficInput):
    try:
        model_input = (
            prepare_prediction_features(data)
        )

        prediction = model.predict(
            model_input
        )[0]

        logger.info(
            "Traffic prediction generated: %.2f",
            prediction,
        )

        return {
            "status": "PASS",
            "predicted_traffic_volume": round(
                float(prediction),
                2,
            ),
        }

    except ValueError as error:
        logger.error(
            "Invalid prediction input: %s",
            error,
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )