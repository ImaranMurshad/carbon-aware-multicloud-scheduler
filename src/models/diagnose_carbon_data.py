
import pandas as pd

DATA_FILE = "data/raw/carbon_history.csv"

df = pd.read_csv(DATA_FILE)

df["datetime"] = pd.to_datetime(
    df["datetime"], utc=True, errors="coerce"
)
df["carbon_intensity"] = pd.to_numeric(
    df["carbon_intensity"], errors="coerce"
)

print("=== DATA QUALITY ===")
print("Total rows:", len(df))
print("Missing timestamps:", df["datetime"].isna().sum())
print("Missing intensity:", df["carbon_intensity"].isna().sum())

for zone in ["IE", "DE", "FR"]:
    data = (
        df[df["zone"] == zone]
        .dropna(subset=["datetime", "carbon_intensity"])
        .sort_values("datetime")
        .drop_duplicates("datetime")
    )

    gaps = data["datetime"].diff().dropna()
    hourly_gaps = (gaps != pd.Timedelta(hours=1)).sum()

    print(f"\n=== {zone} ===")
    print("Rows:", len(data))
    print("Non-hourly intervals:", hourly_gaps)
    print("Minimum intensity:", data["carbon_intensity"].min())
    print("Maximum intensity:", data["carbon_intensity"].max())
    print("Mean intensity:", round(data["carbon_intensity"].mean(), 2))

    if "is_estimated" in data.columns:
        print("Estimation flags:")
        print(data["is_estimated"].value_counts(dropna=False))

    print("Last 12 observations:")
    print(data[["datetime", "carbon_intensity"]].tail(12).to_string(index=False))
