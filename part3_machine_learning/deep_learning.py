import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


logger = logging.getLogger(__name__)


# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# -------------------------------------------------
# Load Part 2 feature-engineered dataset
# -------------------------------------------------
def load_feature_data(csv_path):
    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=["date_time"],
        )

        logger.info(
            "Feature-engineered dataset loaded: "
            "%d rows, %d columns.",
            df.shape[0],
            df.shape[1],
        )

        return df

    except (FileNotFoundError, pd.errors.ParserError):
        logger.error(
            "Unable to load feature-engineered dataset: %s",
            csv_path,
            exc_info=True,
        )
        return None


# -------------------------------------------------
# Prepare the verified 21-feature set
# -------------------------------------------------
def prepare_model_features(df):
    df = df.copy()

    df["holiday_flag"] = (
        df["holiday"]
        .str.lower()
        .ne("none")
        .astype(int)
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    feature_columns = [
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "weekend",
        "holiday_flag",
        "temp_scaled",
        "rain_1h",
        "snow_1h",
        "clouds_all",
    ]

    weather_columns = [
        col for col in df.columns
        if col.startswith("weather_")
        and col not in [
            "weather_main",
            "weather_description",
        ]
    ]

    feature_columns.extend(weather_columns)

    logger.info(
        "Neural-network feature set prepared "
        "with %d features.",
        len(feature_columns),
    )

    logger.debug(
        "Neural-network features: %s",
        feature_columns,
    )

    return df, feature_columns


# -------------------------------------------------
# Neural network
# -------------------------------------------------
class TrafficVolumeNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


# -------------------------------------------------
# Train and evaluate neural network
# -------------------------------------------------
def train_neural_network(df, feature_columns):
    X = df[feature_columns].to_numpy(
        dtype=np.float32
    )

    y = df["traffic_volume"].to_numpy(
        dtype=np.float32
    ).reshape(-1, 1)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    logger.info(
        "Deep-learning data split: "
        "%d training records, %d test records.",
        len(X_train),
        len(X_test),
    )

    # Scale X using training data only
    x_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(
        X_train
    )

    X_test_scaled = x_scaler.transform(
        X_test
    )

    # Scale y separately
    y_scaler = StandardScaler()

    y_train_scaled = y_scaler.fit_transform(
        y_train
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        "PyTorch device: %s",
        device,
    )

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train_scaled,
        dtype=torch.float32,
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
    )

    model = TrafficVolumeNetwork(
        input_size=len(feature_columns)
    ).to(device)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    epochs = 50

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = loss_function(
                predictions,
                batch_y,
            )

            loss.backward()
            optimizer.step()

            epoch_loss += (
                loss.item() * batch_X.size(0)
            )

        average_loss = (
            epoch_loss / len(train_dataset)
        )

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
        ):
            logger.info(
                "Epoch %d/%d -> training loss: %.6f",
                epoch + 1,
                epochs,
                average_loss,
            )

    # ---------------------------------------------
    # Test-set prediction
    # ---------------------------------------------
    model.eval()

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():
        predictions_scaled = (
            model(X_test_tensor)
            .cpu()
            .numpy()
        )

    predictions = y_scaler.inverse_transform(
        predictions_scaled
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    logger.info(
        "Neural Network Regression results:"
    )

    logger.info(
        "MAE: %.2f",
        mae,
    )

    logger.info(
        "R-squared: %.4f",
        r2,
    )

    metrics = {
        "mae": mae,
        "r2": r2,
    }

    return (
        model,
        metrics,
        x_scaler,
        y_scaler,
    )


# -------------------------------------------------
# Save trained model and preprocessing objects
# -------------------------------------------------
def save_model_artifacts(
    model,
    x_scaler,
    y_scaler,
    feature_columns,
):
    models_dir = Path(
        "part3_machine_learning/models"
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        models_dir
        / "neural_network_regression.pt"
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "input_size": len(feature_columns),
            "feature_columns": feature_columns,
        },
        model_path,
    )

    x_scaler_path = (
        models_dir / "nn_x_scaler.joblib"
    )

    y_scaler_path = (
        models_dir / "nn_y_scaler.joblib"
    )

    joblib.dump(
        x_scaler,
        x_scaler_path,
    )

    joblib.dump(
        y_scaler,
        y_scaler_path,
    )

    logger.info(
        "Neural-network model saved: %s",
        model_path,
    )

    logger.info(
        "Input scaler saved: %s",
        x_scaler_path,
    )

    logger.info(
        "Target scaler saved: %s",
        y_scaler_path,
    )


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    csv_path = Path(
        "data/processed/traffic_features_part2.csv"
    )

    df = load_feature_data(csv_path)

    if df is None:
        sys.exit(1)

    df, feature_columns = (
        prepare_model_features(df)
    )

    (
        model,
        metrics,
        x_scaler,
        y_scaler,
    ) = train_neural_network(
        df,
        feature_columns,
    )

    save_model_artifacts(
        model,
        x_scaler,
        y_scaler,
        feature_columns,
    )


# -------------------------------------------------
# Logging configuration
# -------------------------------------------------
if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - "
        "%(funcName)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        "part3_machine_learning/part3.log"
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    main()