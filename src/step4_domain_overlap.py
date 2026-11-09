import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from step1_feature_eng import build_feature_matrix
from step2_primary_rf import run_loso_cv, run_permutation_test, N_PERMUTATIONS
from step3_shap_stability import compute_shap_values, FEATURE_COLS, FEATURE_LABELS

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTCOMES = ["ISI", "PHQ9", "GAD7", "Composite"]
OUTCOME_NAMES = {
    "ISI": "Insomnia (ISI)",
    "PHQ9": "Depression (PHQ-9)",
    "GAD7": "Anxiety (GAD-7)",
    "Composite": "Distress Composite"
}

def run_step4(df=None):
    print("==================================================")
    print("STEP 4: Cross-Domain Signature Overlap")
    print("==================================================")
    
    if df is None:
        df = build_feature_matrix(window_days=14)

    results_summary = []
    shap_matrices = {}

    for outcome in OUTCOMES:
        print(f"\nProcessing outcome: {OUTCOME_NAMES[outcome]}...")
        r2, mae, rmse, actuals, preds = run_loso_cv(df, target_col=outcome)
        _, p_val, _ = run_permutation_test(df, target_col=outcome, n_permutations=N_PERMUTATIONS)
        rf, explainer, shap_vals, shap_df = compute_shap_values(df, target_col=outcome)

        shap_matrices[outcome] = shap_df.set_index("Feature_Raw")["Mean_Abs_SHAP"]

        top_feature = shap_df.iloc[0]["Feature"]
        top_shap_val = shap_df.iloc[0]["Mean_Abs_SHAP"]

        results_summary.append({
            "Outcome": OUTCOME_NAMES[outcome],
            "LOSO_R2": f"{r2:.4f}",
            "LOSO_MAE": f"{mae:.4f}",
            "LOSO_RMSE": f"{rmse:.4f}",
            "Permutation_p": f"{p_val:.4f}",
            "Top_Feature": f"{top_feature} ({top_shap_val:.3f})"
        })

    df_table3 = pd.DataFrame(results_summary)
    t3_path = os.path.join(RESULTS_DIR, "Table3_Cross_Domain_Overlap_Performance.csv")
    df_table3.to_csv(t3_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved Table 3 to: {t3_path}")
    print(df_table3.to_string(index=False))

    df_shap_matrix = pd.DataFrame(shap_matrices)
    df_shap_matrix.index = [FEATURE_LABELS.get(f, f) for f in df_shap_matrix.index]
    df_shap_matrix.columns = [OUTCOME_NAMES[c] for c in df_shap_matrix.columns]

    print("\nMean Absolute SHAP Feature Attributions Across Outcomes:")
    print(df_shap_matrix.round(4))

    rank_df = df_shap_matrix.rank(ascending=False)
    print("\nFeature Importance Rankings (1 = Top Feature):")
    print(rank_df)

    isi_ranks = rank_df["Insomnia (ISI)"]
    phq_ranks = rank_df["Depression (PHQ-9)"]
    gad_ranks = rank_df["Anxiety (GAD-7)"]
    comp_ranks = rank_df["Distress Composite"]

    r_phq, p_phq = spearmanr(isi_ranks, phq_ranks)
    r_gad, p_gad = spearmanr(isi_ranks, gad_ranks)
    r_comp, p_comp = spearmanr(isi_ranks, comp_ranks)

    print(f"\nSpearman Rank Correlation with ISI Signature "
          f"(note: only {len(isi_ranks)} features, so these are very low-powered):")
    print(f"ISI vs PHQ-9: r = {r_phq:.3f} (p = {p_phq:.3f})")
    print(f"ISI vs GAD-7: r = {r_gad:.3f} (p = {p_gad:.3f})")
    print(f"ISI vs Composite: r = {r_comp:.3f} (p = {p_comp:.3f})")

    rank_corr_df = pd.DataFrame([
        {"Comparison": "ISI vs PHQ-9", "Spearman_r": round(r_phq, 3), "p": round(p_phq, 3)},
        {"Comparison": "ISI vs GAD-7", "Spearman_r": round(r_gad, 3), "p": round(p_gad, 3)},
        {"Comparison": "ISI vs Distress Composite", "Spearman_r": round(r_comp, 3), "p": round(p_comp, 3)},
    ])
    rank_corr_df["n_features"] = len(isi_ranks)
    rc_path = os.path.join(RESULTS_DIR, "Table3b_SHAP_Rank_Correlations.csv")
    rank_corr_df.to_csv(rc_path, index=False, encoding="utf-8-sig")
    print(f"Saved Table 3b to: {rc_path}")

    shap_matrix_path = os.path.join(RESULTS_DIR, "Table3c_CrossDomain_SHAP_Matrix.csv")
    df_shap_matrix.round(4).to_csv(shap_matrix_path, encoding="utf-8-sig")
    print(f"Saved Table 3c to: {shap_matrix_path}")

    sns.set_theme(style="white", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_norm = df_shap_matrix / df_shap_matrix.max(axis=0)
    
    sns.heatmap(df_norm, annot=df_shap_matrix.values, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Normalized SHAP Attribution"}, ax=ax, linewidths=0.5)
    ax.set_title("Figure 6. Comparative SHAP Feature Attributions Across Symptom Domains", fontweight="bold", pad=15)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    
    fig6_path = os.path.join(RESULTS_DIR, "Figure6_Cross_Domain_SHAP_Heatmap.png")
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"Saved Figure 6 to: {fig6_path}")

    return df_table3, df_shap_matrix

if __name__ == "__main__":
    run_step4()
