import os
import pandas as pd
import numpy as np
from step1_feature_eng import build_feature_matrix
from step2_primary_rf import run_loso_cv, run_permutation_test, N_PERMUTATIONS
from step3_shap_stability import compute_shap_values, FEATURE_COLS, FEATURE_LABELS

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_step5():
    print("==================================================")
    print("STEP 5: Feature-Window Sensitivity (14-day vs. 7-day)")
    print("==================================================")
    
    df14 = build_feature_matrix(window_days=14)
    df7 = build_feature_matrix(window_days=7)
    
    print("\nRunning Primary ISI Model on 14-day Window Matrix...")
    r2_14, mae_14, rmse_14, _, _ = run_loso_cv(df14, target_col="ISI")
    _, p_14, _ = run_permutation_test(df14, target_col="ISI", n_permutations=N_PERMUTATIONS)
    _, _, _, shap_14 = compute_shap_values(df14, target_col="ISI")

    print("\nRunning Primary ISI Model on 7-day Window Matrix...")
    r2_7, mae_7, rmse_7, _, _ = run_loso_cv(df7, target_col="ISI")
    _, p_7, _ = run_permutation_test(df7, target_col="ISI", n_permutations=N_PERMUTATIONS)
    _, _, _, shap_7 = compute_shap_values(df7, target_col="ISI")
    
    top_14 = f"{shap_14.iloc[0]['Feature']} ({shap_14.iloc[0]['Mean_Abs_SHAP']:.3f})"
    top_7 = f"{shap_7.iloc[0]['Feature']} ({shap_7.iloc[0]['Mean_Abs_SHAP']:.3f})"
    
    top2_14 = f"{shap_14.iloc[1]['Feature']} ({shap_14.iloc[1]['Mean_Abs_SHAP']:.3f})"
    top2_7 = f"{shap_7.iloc[1]['Feature']} ({shap_7.iloc[1]['Mean_Abs_SHAP']:.3f})"
    
    table4_data = [
        {"Metric / Feature": "Analytic Sample Size (N observations)", "14-Day Baseline Window": len(df14), "7-Day Baseline Window": len(df7)},
        {"Metric / Feature": "LOSO-CV R²", "14-Day Baseline Window": f"{r2_14:.4f}", "7-Day Baseline Window": f"{r2_7:.4f}"},
        {"Metric / Feature": "LOSO-CV MAE", "14-Day Baseline Window": f"{mae_14:.4f}", "7-Day Baseline Window": f"{mae_7:.4f}"},
        {"Metric / Feature": "LOSO-CV RMSE", "14-Day Baseline Window": f"{rmse_14:.4f}", "7-Day Baseline Window": f"{rmse_7:.4f}"},
        {"Metric / Feature": f"Permutation Test p-value ({N_PERMUTATIONS} iter.)", "14-Day Baseline Window": f"{p_14:.4f}", "7-Day Baseline Window": f"{p_7:.4f}"},
        {"Metric / Feature": "Top SHAP Feature (#1)", "14-Day Baseline Window": top_14, "7-Day Baseline Window": top_7},
        {"Metric / Feature": "Second SHAP Feature (#2)", "14-Day Baseline Window": top2_14, "7-Day Baseline Window": top2_7},
    ]
    
    df_table4 = pd.DataFrame(table4_data)
    t4_path = os.path.join(RESULTS_DIR, "Table4_Window_Sensitivity_Comparison.csv")
    df_table4.to_csv(t4_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved Table 4 to: {t4_path}")
    print(df_table4.to_string(index=False))
    
    return df_table4

if __name__ == "__main__":
    run_step5()
