import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_survey_data

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_step0():
    print("==================================================")
    print("STEP 0: Sample Description & Table 1")
    print("==================================================")
    
    survey = load_survey_data()
    N = len(survey)
    
    # 1. Demographics
    age_mean, age_std = survey["age"].mean(), survey["age"].std()
    age_min, age_max = survey["age"].min(), survey["age"].max()
    female_cnt = (survey["sex"] == 2).sum()
    female_pct = female_cnt / N * 100
    
    bmi_mean, bmi_std = survey["bmi"].mean(), survey["bmi"].std()
    bmi_n = survey["bmi"].notna().sum()
    
    occ_counts = survey["occupation"].value_counts().to_dict()
    # 1: Undergrad, 2: Grad, 3: Office
    occ_str = f"{occ_counts.get(1, 0)} / {occ_counts.get(2, 0)} / {occ_counts.get(3, 0)}"
    
    meq_mean, meq_std = survey["MEQ"].mean(), survey["MEQ"].std()
    
    # Questionnaire means and SDs
    scales = ["ISI", "PHQ9", "GAD7"]
    t_scores = {}
    for s in scales:
        t_scores[s] = {
            "T1": (survey[f"{s}_1"].mean(), survey[f"{s}_1"].std()),
            "T2": (survey[f"{s}_2"].mean(), survey[f"{s}_2"].std()),
            "T3": (survey[f"{s}_F"].mean(), survey[f"{s}_F"].std()),
        }
    
    # Proportion above cutoff (ISI >= 8)
    all_isi = pd.concat([survey["ISI_1"], survey["ISI_2"], survey["ISI_F"]])
    all_phq = pd.concat([survey["PHQ9_1"], survey["PHQ9_2"], survey["PHQ9_F"]])
    all_gad = pd.concat([survey["GAD7_1"], survey["GAD7_2"], survey["GAD7_F"]])
    
    isi_subclinical_pct = (all_isi >= 8).mean() * 100
    phq_mild_pct = (all_phq >= 5).mean() * 100
    gad_mild_pct = (all_gad >= 5).mean() * 100
    
    # Within-person change (shift >= 3 points in ISI)
    shift_1_2 = (survey["ISI_2"] - survey["ISI_1"]).abs() >= 3
    shift_2_F = (survey["ISI_F"] - survey["ISI_2"]).abs() >= 3
    shift_1_F = (survey["ISI_F"] - survey["ISI_1"]).abs() >= 3
    any_shift = (shift_1_2 | shift_2_F | shift_1_F).sum()
    
    # Inter-questionnaire correlations (pooled across time points)
    df_pooled = pd.DataFrame({
        "ISI": all_isi,
        "PHQ9": all_phq,
        "GAD7": all_gad
    })
    corr_matrix = df_pooled.corr()
    print("Pooled Inter-questionnaire Correlations:")
    print(corr_matrix.round(3))
    
    # Construct Table 1 DataFrame
    table1_rows = [
        {"Characteristic": "N", "Value": f"{N}"},
        {"Characteristic": "Age, years (M ± SD; range)", "Value": f"{age_mean:.1f} ± {age_std:.1f}; {age_min:.0f}-{age_max:.0f}"},
        {"Characteristic": "Female, n (%)", "Value": f"{female_cnt} ({female_pct:.0f}%)"},
        {"Characteristic": "BMI (M ± SD; n)", "Value": f"{bmi_mean:.1f} ± {bmi_std:.1f}; n = {bmi_n} (11 missing)"},
        {"Characteristic": "Occupation (Undergrad / Grad / Office)", "Value": occ_str},
        {"Characteristic": "ISI - T1 / T2 / T3 (M ± SD)", "Value": f"{t_scores['ISI']['T1'][0]:.1f} ± {t_scores['ISI']['T1'][1]:.1f} / {t_scores['ISI']['T2'][0]:.1f} ± {t_scores['ISI']['T2'][1]:.1f} / {t_scores['ISI']['T3'][0]:.1f} ± {t_scores['ISI']['T3'][1]:.1f}"},
        {"Characteristic": "PHQ-9 - T1 / T2 / T3 (M ± SD)", "Value": f"{t_scores['PHQ9']['T1'][0]:.1f} ± {t_scores['PHQ9']['T1'][1]:.1f} / {t_scores['PHQ9']['T2'][0]:.1f} ± {t_scores['PHQ9']['T2'][1]:.1f} / {t_scores['PHQ9']['T3'][0]:.1f} ± {t_scores['PHQ9']['T3'][1]:.1f}"},
        {"Characteristic": "GAD-7 - T1 / T2 / T3 (M ± SD)", "Value": f"{t_scores['GAD7']['T1'][0]:.1f} ± {t_scores['GAD7']['T1'][1]:.1f} / {t_scores['GAD7']['T2'][0]:.1f} ± {t_scores['GAD7']['T2'][1]:.1f} / {t_scores['GAD7']['T3'][0]:.1f} ± {t_scores['GAD7']['T3'][1]:.1f}"},
        {"Characteristic": "MEQ chronotype (M ± SD)", "Value": f"{meq_mean:.1f} ± {meq_std:.1f}"},
        {"Characteristic": "ISI >= 8 (Subclinical Insomnia), % pooled", "Value": f"{isi_subclinical_pct:.1f}%"},
        {"Characteristic": "Participants with ISI shift >= 3 points", "Value": f"{any_shift} of {N} ({any_shift/N*100:.1f}%)"},
        {"Characteristic": "PHQ-9 >= 5 (Mild Depression), % pooled", "Value": f"{phq_mild_pct:.1f}%"},
        {"Characteristic": "GAD-7 >= 5 (Mild Anxiety), % pooled", "Value": f"{gad_mild_pct:.1f}%"},
    ]
    df_table1 = pd.DataFrame(table1_rows)
    table1_path = os.path.join(RESULTS_DIR, "Table1_Sample_Characteristics.csv")
    df_table1.to_csv(table1_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved Table 1 to: {table1_path}")
    
    # 2. Figure 1: Histograms of ISI, PHQ-9, GAD-7
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    titles = ["ISI (Insomnia Severity Index)", "PHQ-9 (Depression)", "GAD-7 (Anxiety)"]
    cutoffs = [8, 5, 5]
    cutoff_labels = ["Subclinical Cutoff (>= 8)", "Mild Cutoff (>= 5)", "Mild Cutoff (>= 5)"]
    data_list = [all_isi, all_phq, all_gad]
    
    for ax, data, color, title, cutoff, clabel in zip(axes, data_list, colors, titles, cutoffs, cutoff_labels):
        sns.histplot(data, ax=ax, color=color, kde=True, discrete=True, alpha=0.6)
        ax.axvline(cutoff, color="red", linestyle="--", linewidth=2, label=clabel)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count (Pooled Observations)")
        ax.legend(loc="upper right")
    
    plt.tight_layout()
    fig1_path = os.path.join(RESULTS_DIR, "Figure1_Symptom_Distributions.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to: {fig1_path}")

if __name__ == "__main__":
    run_step0()
