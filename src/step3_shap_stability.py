import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import shap
from step1_feature_eng import build_feature_matrix

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Kept equal to N_PERMUTATIONS in Step 2 so every resampling figure in the manuscript
# reports the same number of iterations.
N_BOOTSTRAPS = 1000

FEATURE_COLS = [
    "sdnn_adj_dev", "lf/hf_adj_dev", "HR_adj_dev",
    "steps_dev", "sleep_latency_dev", "waso_dev", "sleep_efficiency_dev"
]

FEATURE_LABELS = {
    "sdnn_adj_dev": "SDNN (Daytime Dev)",
    "lf/hf_adj_dev": "LF/HF (Daytime Dev)",
    "HR_adj_dev": "Heart Rate (Daytime Dev)",
    "steps_dev": "Physical Activity (Steps Dev)",
    "sleep_latency_dev": "Sleep Latency Dev",
    "waso_dev": "WASO Dev",
    "sleep_efficiency_dev": "Sleep Efficiency Dev"
}

def compute_shap_values(df, target_col="ISI", feature_cols=FEATURE_COLS, random_state=42):
    X = df[feature_cols]
    y = df[target_col]

    rf = RandomForestRegressor(n_estimators=30, random_state=random_state, max_depth=4, n_jobs=1)
    rf.fit(X, y)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "Feature_Raw": feature_cols,
        "Feature": [FEATURE_LABELS.get(c, c) for c in feature_cols],
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    return rf, explainer, shap_values, shap_df

def compute_bootstrap_stability(df, target_col="ISI", feature_cols=FEATURE_COLS,
                                n_bootstraps=N_BOOTSTRAPS, random_state=42):
    rng = np.random.default_rng(random_state)
    subjects = df["userId"].unique()
    n_subjects = len(subjects)

    feature_ranks = {f: [] for f in feature_cols}
    feature_abs_shaps = {f: [] for f in feature_cols}

    for b in range(n_bootstraps):
        boot_subs = rng.choice(subjects, size=n_subjects, replace=True)
        boot_dfs = [df[df["userId"] == sub] for sub in boot_subs]
        boot_df = pd.concat(boot_dfs, ignore_index=True)

        X_b = boot_df[feature_cols]
        y_b = boot_df[target_col]

        rf = RandomForestRegressor(n_estimators=20, random_state=b, max_depth=4, n_jobs=1)
        rf.fit(X_b, y_b)

        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X_b)
        mean_abs = np.abs(sv).mean(axis=0)

        ranks = len(feature_cols) - np.argsort(np.argsort(mean_abs))

        for idx, f in enumerate(feature_cols):
            feature_ranks[f].append(ranks[idx])
            feature_abs_shaps[f].append(mean_abs[idx])

    stability_rows = []
    for f in feature_cols:
        r_arr = np.array(feature_ranks[f])
        s_arr = np.array(feature_abs_shaps[f])
        stability_rows.append({
            "Feature_Raw": f,
            "Feature": FEATURE_LABELS.get(f, f),
            "Mean_Rank": r_arr.mean(),
            "Median_Rank": np.median(r_arr),
            "Rank_2.5%": np.percentile(r_arr, 2.5),
            "Rank_97.5%": np.percentile(r_arr, 97.5),
            "Mean_Abs_SHAP": s_arr.mean(),
            "SHAP_2.5%": np.percentile(s_arr, 2.5),
            "SHAP_97.5%": np.percentile(s_arr, 97.5)
        })

    stab_df = pd.DataFrame(stability_rows).sort_values("Mean_Rank").reset_index(drop=True)
    return stab_df

def run_step3(df=None):
    print("==================================================")
    print("STEP 3: Interpretable Signature (SHAP) & Stability")
    print("==================================================")
    
    if df is None:
        df = build_feature_matrix(window_days=14)

    rf, explainer, shap_values, shap_df = compute_shap_values(df, target_col="ISI")
    print("\nGlobal SHAP Feature Importance (14-day ISI Model):")
    print(shap_df.to_string(index=False))

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9, 5))
    
    X_renamed = df[FEATURE_COLS].rename(columns=FEATURE_LABELS)
    shap.summary_plot(shap_values, X_renamed, show=False)
    plt.title("Figure 3. Global SHAP Feature Attribution Signature for Insomnia (ISI)", fontweight="bold", pad=15)
    plt.tight_layout()
    fig3_path = os.path.join(RESULTS_DIR, "Figure3_ISI_SHAP_Summary.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3 to: {fig3_path}")

    print(f"\nRunning Bootstrap Rank Stability Analysis ({N_BOOTSTRAPS} resamples)...")
    stab_df = compute_bootstrap_stability(df, target_col="ISI", n_bootstraps=N_BOOTSTRAPS)
    print("\nFeature Rank Stability Table:")
    print(stab_df[["Feature", "Mean_Rank", "Rank_2.5%", "Rank_97.5%", "Mean_Abs_SHAP"]].to_string(index=False))
    
    stab_csv_path = os.path.join(RESULTS_DIR, "Table_Bootstrap_Rank_Stability.csv")
    stab_df.to_csv(stab_csv_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(stab_df))
    means = stab_df["Mean_Rank"]
    lowers = stab_df["Rank_2.5%"]
    uppers = stab_df["Rank_97.5%"]
    
    err_low = means - lowers
    err_high = uppers - means
    
    ax.errorbar(means, y_pos, xerr=[err_low, err_high], fmt="o", color="#2b5c8f", ecolor="#d95f02", elinewidth=2.5, capsize=5, markersize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stab_df["Feature"], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance Rank (1 = Highest SHAP Attribution)")
    ax.set_title(f"Figure 4. Bootstrap Feature Rank Stability ({N_BOOTSTRAPS} Resamples, 95% CI)",
                 fontweight="bold")
    plt.tight_layout()
    fig4_path = os.path.join(RESULTS_DIR, "Figure4_Bootstrap_Rank_Stability.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved Figure 4 to: {fig4_path}")

    subclinical_subs = df[df["ISI"] >= 8]["userId"].unique()
    case_rng = np.random.default_rng(123)
    chosen_subs = case_rng.choice(subclinical_subs, size=2, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for idx, sub in enumerate(chosen_subs):
        sub_rows = df[df["userId"] == sub]
        sub_row = sub_rows.iloc[0]
        # shap_values is positional, so translate the label index before slicing it.
        row_shap = shap_values[df.index.get_loc(sub_rows.index[0])]

        feat_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
            "Value": [sub_row[f] for f in FEATURE_COLS],
            "SHAP": row_shap
        }).sort_values("SHAP", key=abs, ascending=True)
        
        ax = axes[idx]
        colors = ["#d95f02" if s > 0 else "#2b5c8f" for s in feat_df["SHAP"]]
        ax.barh(feat_df["Feature"], feat_df["SHAP"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Participant {sub} (Observed ISI = {sub_row['ISI']})", fontweight="bold")
        ax.set_xlabel("SHAP Impact on Insomnia Prediction")
        
    plt.suptitle("Figure 5. Illustrative Individual SHAP Attributions for Subclinical Cases", fontweight="bold", fontsize=14)
    plt.tight_layout()
    fig5_path = os.path.join(RESULTS_DIR, "Figure5_Individual_SHAP_Cases.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved Figure 5 to: {fig5_path}")

    return shap_df, stab_df

if __name__ == "__main__":
    run_step3()
