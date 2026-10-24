
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error

DATA_FILE = "data/raw/carbon_history.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )
    df["carbon_intensity"] = pd.to_numeric(
        df["carbon_intensity"], errors="coerce"
    )

    for zone in ["IE", "DE", "FR"]:
        data = (
            df[df["zone"] == zone]
            .dropna(subset=["datetime", "carbon_intensity"])
            .sort_values("datetime")
            .drop_duplicates("datetime")
            .set_index("datetime")
        )

        data["target"] = data["carbon_intensity"].shift(-1)
        data["lag_1"] = data["carbon_intensity"]
        data["lag_2"] = data["carbon_intensity"].shift(1)
        data["lag_24"] = data["carbon_intensity"].shift(23)
        data["hour"] = data.index.hour
        data["day_of_week"] = data.index.day_of_week

        features = [
            "lag_1", "lag_2", "lag_24",
            "hour", "day_of_week"
        ]

        data = data.dropna(subset=features + ["target"])

        split = int(len(data) * 0.8)
        test = data.iloc[split:]

        artifact = joblib.load(
            f"models/carbon_model_{zone}.joblib"
        )
        model = artifact["model"]

        predictions = model.predict(test[features])
        actual = test["target"].to_numpy()
        baseline = test["lag_1"].to_numpy()

        print(f"\n{'=' * 45}")
        print(f"ZONE: {zone}")
        print(f"Training target mean: {data.iloc[:split]['target'].mean():.2f}")
        print(f"Test target mean: {test['target'].mean():.2f}")
        print(f"Model prediction mean: {predictions.mean():.2f}")
        print(f"Baseline MAE: {mean_absolute_error(actual, baseline):.2f}")
        print(f"Model MAE: {mean_absolute_error(actual, predictions):.2f}")

        comparison = pd.DataFrame({
            "actual_next_hour": actual,
            "random_forest": predictions,
            "persistence": baseline,
        }, index=test.index)

        print("\nLast 10 test predictions:")
        print(comparison.tail(10).round(2).to_string())


if __name__ == "__main__":
    main()
