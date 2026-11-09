"""
Robustness check for the zero-imputation choice in Step 1.

Observations with no usable daytime HRV have their three HRV deviation features imputed to
zero, i.e. "no deviation from this participant's own baseline". That is defensible but it
injects a fabricated constant into a meaningful fraction of rows. This step re-runs the
primary ISI model on HRV complete cases only, so the reported conclusion can be stated as
robust (or not) to that choice rather than merely assumed to be.
"""
import os
import pandas as pd
import numpy as np
from step1_feature_eng import build_feature_matrix
from step2_primary_rf import (run_loso_cv, run_permutation_test, null_model_benchmarks,
                              N_PERMUTATIONS)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

HRV_COLS = ["sdnn_adj_dev", "lf/hf_adj_dev", "HR_adj_dev"]


def run_step6(df=None):
    print("=" * 50)
    print("STEP 6: Zero-Imputation Sensitivity (HRV complete cases)")
    print("=" * 50)

    if df is None:
        df = build_feature_matrix(window_days=14)

    imputed_mask = (df[HRV_COLS] == 0).all(axis=1)
    df_cc = df.loc[~imputed_mask].reset_index(drop=True)

    print(f"Full sample:          {len(df)} observations, {df['userId'].nunique()} subjects")
    print(f"HRV complete cases:   {len(df_cc)} observations, {df_cc['userId'].nunique()} subjects")
    print(f"Excluded (no HRV):    {int(imputed_mask.sum())}")

    rows = []
    for label, d in [("All observations (zero-imputed)", df),
                     ("HRV complete cases only", df_cc)]:
        r2, mae, rmse, _, _ = run_loso_cv(d, target_col="ISI")
        _, p_val, _ = run_permutation_test(d, target_col="ISI", n_permutations=N_PERMUTATIONS)
        bench = null_model_benchmarks(d, target_col="ISI")
        print(f"\n{label}: R2 = {r2:.4f}, p = {p_val:.4f}, RMSE = {rmse:.4f}")
        rows.append({
            "Sample": label,
            "N_observations": len(d),
            "N_subjects": d["userId"].nunique(),
            "LOSO_R2": f"{r2:.4f}",
            "LOSO_MAE": f"{mae:.4f}",
            "LOSO_RMSE": f"{rmse:.4f}",
            "Permutation_p": f"{p_val:.4f}",
            "GrandMean_RMSE": f"{bench['grand_mean_RMSE']:.4f}",
            "SubjectMean_RMSE": f"{bench['subject_mean_RMSE']:.4f}",
        })

    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "Table5_Imputation_Sensitivity.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nSaved Table 5 to: {path}")
    print(out.to_string(index=False))

    both_null = all(float(r["LOSO_R2"]) < 0 for r in rows)
    print(f"\nConclusion unchanged by imputation choice (both R2 < 0): {both_null}")
    return out


if __name__ == "__main__":
    run_step6()
