import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"  # Figshare download (doi:10.6084/m9.figshare.28509740), see README
)

def load_survey_data(data_dir=DATA_DIR):
    """Loads survey demographics and scale scores (ISI, PHQ9, GAD7, MEQ)."""
    path = os.path.join(data_dir, "survey.csv")
    df = pd.read_csv(path)
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    return df

def load_sleep_diary(data_dir=DATA_DIR):
    """Loads sleep diary data, converts date column to datetime."""
    path = os.path.join(data_dir, "sleep_diary.csv")
    df = pd.read_csv(path)
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(["userId", "date_dt"]).reset_index(drop=True)
    return df

# The cohort was recorded in South Korea (KST = UTC+9). Unix epoch timestamps decode to UTC,
# so they must be shifted before any hour-of-day or calendar-date logic is applied.
# Verified against two independent signals: the activity trough falls at 23:00-07:00 only on
# the KST axis, and sleep-diary wake times (08:20-11:00) are local by construction.
KST_OFFSET_HOURS = 9


def load_hrv_filtered(data_dir=DATA_DIR):
    """Loads 5-minute pre-filtered sensor HRV data, converted to Korea local time (KST)."""
    path = os.path.join(data_dir, "sensor_hrv_filtered.csv")
    df = pd.read_csv(path)
    # ts_start is Unix milliseconds (UTC); shift to KST so that "hour" means local hour.
    df["datetime_utc"] = pd.to_datetime(df["ts_start"], unit="ms")
    df["datetime"] = df["datetime_utc"] + pd.Timedelta(hours=KST_OFFSET_HOURS)
    # Exclude invalid timestamps outside 2021
    df = df[df["datetime"].dt.year == 2021].copy()
    # date_dt is the local calendar date, which is what the sleep diary is keyed on.
    df["date_dt"] = pd.to_datetime(df["datetime"].dt.date)
    df["hour"] = df["datetime"].dt.hour
    return df

if __name__ == "__main__":
    survey = load_survey_data()
    sleep = load_sleep_diary()
    hrv = load_hrv_filtered()
    print(f"Loaded survey N={len(survey)}, sleep records N={len(sleep)}, HRV slices N={len(hrv)}")
