import os
import sys

# Ensure src/ directory is in python module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step0_sample_desc import run_step0
from step1_feature_eng import build_feature_matrix
from step2_primary_rf import run_step2
from step3_shap_stability import run_step3
from step4_domain_overlap import run_step4
from step5_sensitivity import run_step5
from step6_imputation_sensitivity import run_step6
from step7_cooccurrence import run_step7

def run_pipeline():
    print("================================================================")
    print("STARTING FULL ANALYSIS PIPELINE: DIGITAL PHENOTYPING INSOMNIA")
    print("================================================================")

    # Step 0: Sample Description & Table 1
    run_step0()

    # Step 1: Feature Engineering
    df14 = build_feature_matrix(window_days=14)
    df7 = build_feature_matrix(window_days=7)

    # Step 2: Primary Model (Insomnia / ISI)
    df14, r2_isi, p_isi = run_step2(df14)

    # Step 3: SHAP Interpretability & Bootstrap Stability
    shap_isi, stab_isi = run_step3(df14)

    # Step 4: Cross-Domain Overlap (PHQ-9, GAD-7, Composite)
    df_t3, df_shap_matrix = run_step4(df14)

    # Step 5: Feature-Window Sensitivity (14-day vs 7-day)
    df_t4 = run_step5()

    # Step 6: Zero-Imputation Sensitivity (HRV complete cases)
    df_t5 = run_step6(df14)

    # Step 7: Symptom co-occurrence (descriptive, Table 6)
    run_step7()

    print("================================================================")
    print("PIPELINE EXECUTION COMPLETE! ALL TABLES AND FIGURES GENERATED.")
    print("================================================================")

if __name__ == "__main__":
    run_pipeline()
