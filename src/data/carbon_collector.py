
import os
import csv
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ELECTRICITY_MAPS_API_KEY")
BASE_URL = "https://api.electricitymaps.com/v4"
OUTPUT_FILE = Path("data/raw/carbon_history.csv")

ZONES = {
    "IE": "AWS",
    "DE": "Azure-local",
    "FR": "GCP-local",
}


def fetch_carbon_history(zone):
    if not API_KEY:
        raise RuntimeError("Missing Electricity Maps API key")

    url = f"{BASE_URL}/carbon-intensity/history"

    response = requests.get(
        url,
        params={"zone": zone},
        headers={"auth-token": API_KEY},
        timeout=30,
    )

    response.raise_for_status()
    payload = response.json()

    rows = []

    for item in payload.get("history", []):
        intensity = item.get(
            "carbonIntensity",
            item.get("value"),
        )

        if intensity is None:
            continue

        rows.append({
            "datetime": item["datetime"],
            "zone": zone,
            "environment": ZONES[zone],
            "carbon_intensity": intensity,
            "is_estimated": item.get("isEstimated"),
            "source": "Electricity Maps",
        })

    return rows



def save_to_csv(rows):
    import pandas as pd

    if not rows:
        print("No new observations to save.")
        return

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_data = pd.DataFrame(rows)

    if OUTPUT_FILE.exists():
        existing_data = pd.read_csv(OUTPUT_FILE)
        combined = pd.concat(
            [existing_data, new_data],
            ignore_index=True,
        )
    else:
        combined = new_data

    combined["datetime"] = pd.to_datetime(
        combined["datetime"],
        utc=True,
        errors="coerce",
    )

    combined["carbon_intensity"] = pd.to_numeric(
        combined["carbon_intensity"],
        errors="coerce",
    )

    combined = combined.dropna(
        subset=["datetime", "zone", "carbon_intensity"]
    )

    combined = combined.drop_duplicates(
        subset=["zone", "datetime"],
        keep="last",
    )

    combined = combined.sort_values(
        ["zone", "datetime"]
    )

    combined["datetime"] = (
        combined["datetime"].dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"Total unique observations: {len(combined)}")
    print(f"Saved to: {OUTPUT_FILE}")

    if not rows:
        return

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    all_rows = []

    for zone, environment in ZONES.items():
        try:
            rows = fetch_carbon_history(zone)
            all_rows.extend(rows)

            print(
                f"{environment} ({zone}): "
                f"{len(rows)} observations"
            )

        except requests.RequestException as error:
            print(f"Failed for {zone}: {error}")

    save_to_csv(all_rows)

    print(f"Total observations: {len(all_rows)}")

    if all_rows:
        print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
