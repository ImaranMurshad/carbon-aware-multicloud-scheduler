
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_FILE = Path("data/raw/carbon_history.csv")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")


def calculate_metrics(actual, predicted):
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
    }


def prepare_zone_data(df, zone):
    data = df[df["zone"] == zone].copy()

    data["datetime"] = pd.to_datetime(
        data["datetime"], utc=True, errors="coerce"
    )
    data["carbon_intensity"] = pd.to_numeric(
        data["carbon_intensity"], errors="coerce"
    )

    data = (
        data.dropna(subset=["datetime", "carbon_intensity"])
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .set_index("datetime")
    )

    # Predict the next recorded hour.
    data["target"] = data["carbon_intensity"].shift(-1)

    # Features use only information available at prediction time.
    data["lag_1"] = data["carbon_intensity"]
    data["lag_2"] = data["carbon_intensity"].shift(1)
    data["lag_24"] = data["carbon_intensity"].shift(23)
    data["hour"] = data.index.hour
    data["day_of_week"] = data.index.day_of_week

    feature_columns = [
        "lag_1",
        "lag_2",
        "lag_24",
        "hour",
        "day_of_week",
    ]

    data = data.dropna(
        subset=feature_columns + ["target"]
    )

    return data, feature_columns


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for zone in ["IE", "DE", "FR"]:
        data, features = prepare_zone_data(df, zone)

        if len(data) < 50:
            print(
                f"Skipping {zone}: only {len(data)} usable rows."
            )
            continue

        # Chronological split: earlier observations train,
        # later observations test.
        split = int(len(data) * 0.8)

        train = data.iloc[:split]
        test = data.iloc[split:]

        X_train = train[features]
        y_train = train["target"]

        X_test = test[features]
        y_test = test["target"]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Persistence baseline: next value equals current value.
        baseline_predictions = test["lag_1"].to_numpy()

        model_metrics = calculate_metrics(
            y_test, predictions
        )
        baseline_metrics = calculate_metrics(
            y_test, baseline_predictions
        )

        model_path = MODEL_DIR / f"carbon_model_{zone}.joblib"
        joblib.dump(
            {
                "model": model,
                "features": features,
                "zone": zone,
                "forecast_horizon": "next recorded hour",
            },
            model_path,
        )

        results.append({
            "zone": zone,
            "train_rows": len(train),
            "test_rows": len(test),
            "model_mae": model_metrics["mae"],
            "model_rmse": model_metrics["rmse"],
            "baseline_mae": baseline_metrics["mae"],
            "baseline_rmse": baseline_metrics["rmse"],
        })

        print(f"\nZone: {zone}")
        print(f"Training rows: {len(train)}")
        print(f"Testing rows: {len(test)}")
        print(
            f"Random Forest MAE: "
            f"{model_metrics['mae']:.2f}"
        )
        print(
            f"Persistence MAE: "
            f"{baseline_metrics['mae']:.2f}"
        )
        print(
            f"Random Forest RMSE: "
            f"{model_metrics['rmse']:.2f}"
        )
        print(
            f"Persistence RMSE: "
            f"{baseline_metrics['rmse']:.2f}"
        )
        print(f"Model saved to: {model_path}")

    if results:
        results_file = RESULTS_DIR / "prediction_metrics.csv"
        pd.DataFrame(results).to_csv(
            results_file, index=False
        )
        print(f"\nMetrics saved to: {results_file}")
    else:
        print("No models were trained.")


if __name__ == "__main__":
    main()
