import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import Parallel, delayed
import statsmodels.formula.api as smf
from step1_feature_eng import build_feature_matrix

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Single source of truth for resampling counts, imported by Steps 4 and 5 so that every
# reported p-value in the manuscript rests on the same number of permutations.
N_PERMUTATIONS = 1000
RANDOM_SEED = 42
# Capped (not -1) so joblib workers stay within available RAM; on this machine an
# unbounded worker pool paged to disk and ran 20-40x slower.
N_JOBS = 4

FEATURE_COLS = [
    "sdnn_adj_dev", "lf/hf_adj_dev", "HR_adj_dev",
    "steps_dev", "sleep_latency_dev", "waso_dev", "sleep_efficiency_dev"
]

FEATURE_LABELS = {
    "sdnn_adj_dev": "SDNN Dev",
    "lf/hf_adj_dev": "LF/HF Dev",
    "HR_adj_dev": "Heart Rate Dev",
    "steps_dev": "Steps Dev",
    "sleep_latency_dev": "Sleep Latency Dev",
    "waso_dev": "WASO Dev",
    "sleep_efficiency_dev": "Sleep Efficiency Dev"
}

def run_loso_cv(df, target_col="ISI", feature_cols=FEATURE_COLS, n_estimators=50, random_state=42):
    subjects = df["userId"].unique()
    preds = np.zeros(len(df))
    actuals = df[target_col].values

    for sub in subjects:
        test_idx = df["userId"] == sub
        train_idx = ~test_idx

        X_train, y_train = df.loc[train_idx, feature_cols], df.loc[train_idx, target_col]
        X_test = df.loc[test_idx, feature_cols]

        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, max_depth=4, n_jobs=1)
        rf.fit(X_train, y_train)
        preds[test_idx] = rf.predict(X_test)

    r2 = r2_score(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    return r2, mae, rmse, actuals, preds

def null_model_benchmarks(df, target_col="ISI"):
    """
    Reference points needed to interpret a LOSO R-squared, which is computed against the
    grand mean of the outcome. With only two observations per subject, between-person
    variance dominates, so the grand mean is a demanding benchmark. We therefore also
    report the error of a subject-specific mean predictor, which is unavailable to a
    LOSO model by construction and represents the ceiling any such model works below.
    """
    y = df[target_col].values
    grand = np.full(len(df), y.mean())
    subj_mean = df.groupby("userId")[target_col].transform("mean").values
    return {
        "grand_mean_MAE": mean_absolute_error(y, grand),
        "grand_mean_RMSE": float(np.sqrt(mean_squared_error(y, grand))),
        "subject_mean_MAE": mean_absolute_error(y, subj_mean),
        "subject_mean_RMSE": float(np.sqrt(mean_squared_error(y, subj_mean))),
        "outcome_SD": float(y.std(ddof=1)),
    }


def _one_permutation(df, target_col, feature_cols, shuffled_y, model_seed):
    df_perm = df.copy()
    df_perm[target_col] = shuffled_y
    r2_p, _, _, _, _ = run_loso_cv(df_perm, target_col, feature_cols, random_state=model_seed)
    return r2_p


def run_permutation_test(df, target_col="ISI", feature_cols=FEATURE_COLS,
                         n_permutations=N_PERMUTATIONS, random_state=RANDOM_SEED,
                         n_jobs=N_JOBS):
    """
    Non-parametric test of out-of-sample performance. Outcome labels are shuffled while the
    model seed is held fixed, so the null distribution reflects only the destroyed
    outcome-feature association and not run-to-run forest variability.
    """
    rng = np.random.default_rng(random_state)
    true_r2, _, _, _, _ = run_loso_cv(df, target_col, feature_cols, random_state=random_state)

    y = df[target_col].values
    shuffles = [rng.permutation(y) for _ in range(n_permutations)]

    perm_r2s = np.array(Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_one_permutation)(df, target_col, feature_cols, s, random_state)
        for s in shuffles
    ))

    # Add-one (biased) estimator: p is never exactly zero, which is required because a
    # permutation p-value is bounded below by 1/(n_permutations + 1).
    p_value = (1.0 + np.sum(perm_r2s >= true_r2)) / (1.0 + n_permutations)
    return true_r2, p_value, perm_r2s

def run_mixed_effects(df, target_col="ISI", feature_cols=FEATURE_COLS):
    df_lme = df.copy()
    df_lme["target_std"] = (df_lme[target_col] - df_lme[target_col].mean()) / df_lme[target_col].std()
    
    formula_parts = []
    for col in feature_cols:
        clean_col = col.replace("/", "_")
        std_col = f"{clean_col}_std"
        df_lme[std_col] = (df_lme[col] - df_lme[col].mean()) / df_lme[col].std()
        formula_parts.append(std_col)
        
    formula = f"target_std ~ {' + '.join(formula_parts)}"
    
    try:
        model = smf.mixedlm(formula, df_lme, groups=df_lme["userId"])
        fit = model.fit()
        fe = fit.fe_params
        bse = fit.bse
        summary_df = pd.DataFrame({
            "Coef": fe,
            "SE": bse,
            "CI_lower": fe - 1.96 * bse,
            "CI_upper": fe + 1.96 * bse,
            "pvalue": fit.pvalues
        })
        return fit, summary_df
    except Exception as e:
        print(f"LME fitting warning: {e}")
        return None, None

def run_step2(df=None):
    print("==================================================")
    print("STEP 2: Primary Model (Insomnia / ISI)")
    print("==================================================")
    
    if df is None:
        df = build_feature_matrix(window_days=14)
        
    r2, mae, rmse, actuals, preds = run_loso_cv(df, target_col="ISI")
    print(f"LOSO-CV Performance (14-day window, N = {len(df)} observations):")
    print(f"Out-of-sample R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    bench = null_model_benchmarks(df, target_col="ISI")
    print("\nNull-model benchmarks (context for a negative R²):")
    for k, v in bench.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nRunning Permutation Test ({N_PERMUTATIONS} iterations, parallel)...")
    true_r2, p_val, perm_r2s = run_permutation_test(df, target_col="ISI")
    print(f"Empirical Permutation p-value: {p_val:.4f}")
    print(f"Permutation null R²: mean = {perm_r2s.mean():.4f}, "
          f"95% range = [{np.percentile(perm_r2s, 2.5):.4f}, {np.percentile(perm_r2s, 97.5):.4f}]")

    lme_fit, lme_table = run_mixed_effects(df, target_col="ISI")
    if lme_table is not None:
        print("\nMixed-Effects Model Standardized Estimates (ISI, subject random intercept):")
        print(lme_table.round(4).to_string())
        lme_path = os.path.join(RESULTS_DIR, "Table2b_ISI_MixedEffects.csv")
        lme_table.round(4).to_csv(lme_path, encoding="utf-8-sig")
        print(f"Saved Table 2b to: {lme_path}")
        n_sig = int((lme_table.drop(index=[i for i in ("Intercept", "Group Var")
                                           if i in lme_table.index])["pvalue"] < 0.05).sum())
        print(f"Fixed effects significant at p < .05: {n_sig} of 7")

    t2_data = [
        {"Metric": "Analytic sample (observations / subjects)",
         "Value": f"{len(df)} / {df['userId'].nunique()}"},
        {"Metric": "LOSO-CV R²", "Value": f"{r2:.4f}"},
        {"Metric": "LOSO-CV MAE", "Value": f"{mae:.4f}"},
        {"Metric": "LOSO-CV RMSE", "Value": f"{rmse:.4f}"},
        {"Metric": "Permutation Test p-value",
         "Value": f"{p_val:.4f} (based on {N_PERMUTATIONS} iterations)"},
        {"Metric": "Permutation null R² (mean)", "Value": f"{perm_r2s.mean():.4f}"},
        {"Metric": "Benchmark: grand-mean predictor RMSE", "Value": f"{bench['grand_mean_RMSE']:.4f}"},
        {"Metric": "Benchmark: subject-mean predictor RMSE", "Value": f"{bench['subject_mean_RMSE']:.4f}"},
        {"Metric": "Outcome SD (ISI)", "Value": f"{bench['outcome_SD']:.4f}"},
    ]
    df_t2 = pd.DataFrame(t2_data)
    t2_path = os.path.join(RESULTS_DIR, "Table2_ISI_Model_Performance.csv")
    df_t2.to_csv(t2_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved Table 2 to: {t2_path}")
    
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.regplot(x=actuals, y=preds, ax=ax1, color="#2b5c8f", scatter_kws={"s": 50, "alpha": 0.7})
    ax1.plot([0, 20], [0, 20], "k--", alpha=0.6, label="Identity line")
    ax1.set_title("LOSO-CV Predicted vs. Observed ISI", fontweight="bold")
    ax1.set_xlabel("Observed ISI Score")
    ax1.set_ylabel("LOSO-CV Predicted ISI Score")
    ax1.legend()
    
    sns.histplot(perm_r2s, ax=ax2, color="gray", kde=True, bins=25, alpha=0.5)
    ax2.axvline(true_r2, color="red", linestyle="--", linewidth=2.5, label=f"Observed R² = {true_r2:.3f} (p = {p_val:.3f})")
    ax2.set_title(f"Permutation Null Distribution ({N_PERMUTATIONS} Shuffles)", fontweight="bold")
    ax2.set_xlabel("LOSO R²")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    
    plt.tight_layout()
    fig2_path = os.path.join(RESULTS_DIR, "Figure2_ISI_Model_Validation.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to: {fig2_path}")
    
    return df, r2, p_val

if __name__ == "__main__":
    run_step2()
