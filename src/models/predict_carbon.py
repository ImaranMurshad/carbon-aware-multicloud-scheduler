
from pathlib import Path

import pandas as pd
import joblib

DATA_FILE = Path("data/raw/carbon_history.csv")
MODEL_DIR = Path("models")


def predict_next_hour(zone: str) -> dict:
    """Forecast the next hourly value using the validated baseline."""

    df = pd.read_csv(DATA_FILE)

    df = df[df["zone"] == zone].copy()

    if df.empty:
        raise ValueError(f"No carbon data found for zone {zone}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )
    df["carbon_intensity"] = pd.to_numeric(
        df["carbon_intensity"], errors="coerce"
    )

    df = (
        df.dropna(subset=["datetime", "carbon_intensity"])
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
    )

    latest = df.iloc[-1]

    return {
        "zone": zone,
        "last_observation_time": latest["datetime"].isoformat(),
        "last_observed_intensity": float(
            latest["carbon_intensity"]
        ),
        "predicted_next_intensity": float(
            latest["carbon_intensity"]
        ),
        "method": "persistence_baseline",
    }


if __name__ == "__main__":
    for zone in ["IE", "DE", "FR"]:
        result = predict_next_hour(zone)
        print(result)
