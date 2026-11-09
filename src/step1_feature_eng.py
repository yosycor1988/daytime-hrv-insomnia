import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from data_loader import load_survey_data, load_sleep_diary, load_hrv_filtered

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Daytime window, in Korea local time (see KST_OFFSET_HOURS in data_loader).
# Inclusive hour buckets: 8..22 == 08:00:00-22:59:59.
DAY_START_HOUR = 8
DAY_END_HOUR = 22
DAY_WINDOW_LABEL = "08:00-23:00 local (KST)"

def residualize_hrv_slice(hrv_df):
    """
    Adjusts daytime 5-minute HRV features (sdnn, lf/hf, HR) for concurrent physical activity (steps)
    and time of day (hour) using OLS linear regression residuals.
    """
    df = hrv_df.copy()
    # Daytime window in Korea local time. hour is an integer bucket, so `<= DAY_END_HOUR`
    # retains that whole hour: 8..22 spans 08:00:00-22:59:59, reported as 08:00-23:00.
    df = df[(df["hour"] >= DAY_START_HOUR) & (df["hour"] <= DAY_END_HOUR)].copy()

    feature_cols = ["sdnn", "lf/hf", "HR"]
    for col in feature_cols:
        if col in df.columns:
            valid_mask = df[col].notna() & df["steps"].notna() & df["hour"].notna()
            if valid_mask.sum() > 20:
                X = sm.add_constant(df.loc[valid_mask, ["steps", "hour"]])
                y = df.loc[valid_mask, col]
                model = sm.OLS(y, X).fit()
                df.loc[valid_mask, f"{col}_adj"] = model.resid + y.mean()
            else:
                df[f"{col}_adj"] = df[col]
    return df

def aggregate_daily_features(hrv_df, sleep_df):
    """
    Aggregates daily median of residualized daytime HRV + daily activity + daily sleep diary metrics.
    """
    hrv_adj = residualize_hrv_slice(hrv_df)

    daily_hrv = hrv_adj.groupby(["deviceId", "date_dt"]).agg({
        "sdnn_adj": "median",
        "lf/hf_adj": "median",
        "HR_adj": "median",
        "steps": "sum"
    }).reset_index()

    daily_hrv.rename(columns={"deviceId": "userId"}, inplace=True)

    sleep_cols = ["userId", "date_dt", "sleep_latency", "waso", "sleep_efficiency"]
    daily_combined = pd.merge(daily_hrv, sleep_df[sleep_cols], on=["userId", "date_dt"], how="outer")
    daily_combined.sort_values(["userId", "date_dt"], inplace=True)
    return daily_combined

def build_feature_matrix(window_days=14):
    """
    Builds individualized deviation feature matrix for T2 and T3 questionnaire follow-ups.

    All observations with complete questionnaire outcomes are retained (no truncation to a
    pre-specified N), so the 14-day and 7-day windows are evaluated on the same observations
    and the window comparison in Step 5 is not confounded by differing sample sizes.
    """
    survey = load_survey_data()
    sleep = load_sleep_diary()
    hrv = load_hrv_filtered()

    daily_df = aggregate_daily_features(hrv, sleep)

    dates = sorted(daily_df["date_dt"].dropna().unique())

    if window_days == 14:
        t2_dates = dates[:14]
        t3_dates = dates[14:28]
    elif window_days == 7:
        t2_dates = dates[7:14]
        t3_dates = dates[21:28]
    else:
        raise ValueError("window_days must be 14 or 7")

    feature_cols = ["sdnn_adj", "lf/hf_adj", "HR_adj", "steps", "sleep_latency", "waso", "sleep_efficiency"]

    rows = []
    for _, row in survey.iterrows():
        uid = row["deviceId"]
        user_daily = daily_df[daily_df["userId"] == uid]

        for timepoint, t_dates, isi_val, phq_val, gad_val in [
            ("T2", t2_dates, row["ISI_2"], row["PHQ9_2"], row["GAD7_2"]),
            ("T3", t3_dates, row["ISI_F"], row["PHQ9_F"], row["GAD7_F"])
        ]:
            sub_window = user_daily[user_daily["date_dt"].isin(t_dates)]

            means = sub_window[feature_cols].mean()
            stds = sub_window[feature_cols].std().replace(0, np.nan)

            recent_sub = sub_window.tail(3)
            recent_means = recent_sub[feature_cols].mean()

            dev = recent_means - means
            z_scores = (recent_means - means) / stds

            record = {
                "userId": uid,
                "timepoint": timepoint,
                "ISI": isi_val,
                "PHQ9": phq_val,
                "GAD7": gad_val,
                "Composite": np.nan,
                "n_days_window": len(sub_window)
            }

            for f in feature_cols:
                record[f"{f}_dev"] = dev[f]
                record[f"{f}_z"] = z_scores[f]
                record[f"{f}_mean"] = means[f]

            rows.append(record)

    matrix = pd.DataFrame(rows)

    # Drop observations with a missing questionnaire outcome. reset_index keeps label and
    # positional indices aligned, which downstream SHAP row lookups rely on.
    matrix.dropna(subset=["ISI", "PHQ9", "GAD7"], inplace=True)
    matrix.reset_index(drop=True, inplace=True)

    # Impute 0 for deviation features where a participant had no wearable data in the window
    # (0 = no deviation from that participant's own baseline). Counted and reported, not silent.
    dev_cols = [f"{f}_dev" for f in feature_cols]
    n_imputed = int(matrix[dev_cols].isna().sum().sum())
    imputed_by_feature = matrix[dev_cols].isna().sum()
    rows_no_hrv = int(matrix[["sdnn_adj_dev", "lf/hf_adj_dev", "HR_adj_dev"]].isna().all(axis=1).sum())
    matrix[dev_cols] = matrix[dev_cols].fillna(0)

    print(f"  [{window_days}d] deviation cells imputed to 0: {n_imputed} of {matrix[dev_cols].size} "
          f"({n_imputed / matrix[dev_cols].size * 100:.1f}%)")
    print(f"  [{window_days}d] observations with no usable daytime HRV: {rows_no_hrv} of {len(matrix)}")
    print(f"  [{window_days}d] imputed per feature: "
          f"{ {k: int(v) for k, v in imputed_by_feature.items() if v > 0} }")

    # Standardize scale scores to compute composite distress score (ISI_std + PHQ9_std + GAD7_std)
    matrix["ISI_std"] = (matrix["ISI"] - matrix["ISI"].mean()) / matrix["ISI"].std()
    matrix["PHQ9_std"] = (matrix["PHQ9"] - matrix["PHQ9"].mean()) / matrix["PHQ9"].std()
    matrix["GAD7_std"] = (matrix["GAD7"] - matrix["GAD7"].mean()) / matrix["GAD7"].std()
    matrix["Composite"] = matrix["ISI_std"] + matrix["PHQ9_std"] + matrix["GAD7_std"]

    out_file = os.path.join(RESULTS_DIR, f"Feature_Matrix_{window_days}d.csv")
    matrix.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"Saved {window_days}-day Feature Matrix to {out_file} (Shape: {matrix.shape})")
    return matrix

if __name__ == "__main__":
    print("==================================================")
    print("STEP 1: Feature Engineering")
    print("==================================================")
    df14 = build_feature_matrix(window_days=14)
    df7 = build_feature_matrix(window_days=7)
    print(f"14-day window rows: {len(df14)}")
    print(f"7-day window rows: {len(df7)}")
