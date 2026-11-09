"""
Descriptive symptom co-occurrence across the three domains (insomnia, depression, anxiety).

This is a characterization step, not an inferential one. Its purpose is to quantify how far
the three questionnaires flag the same people, which is the concrete justification for the
interpretive guardrail used in Step 4: overlapping SHAP rankings across outcomes reflect
shared distress variance rather than disorder-specific physiology.

Cutoffs follow the instruments' published thresholds (dataset README):
    ISI   >= 8  subthreshold insomnia
    PHQ-9 >= 5  mild depression
    GAD-7 >= 5  mild anxiety
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from data_loader import load_survey_data

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

CUTOFFS = {"ISI": 8, "PHQ9": 5, "GAD7": 5}
LABELS = {"ISI": "Insomnia\n(ISI $\\geq$ 8)",
          "PHQ9": "Depression\n(PHQ-9 $\\geq$ 5)",
          "GAD7": "Anxiety\n(GAD-7 $\\geq$ 5)"}
COLORS = {"ISI": "#2b5c8f", "PHQ9": "#d95f02", "GAD7": "#7570b3"}
SUFFIX = {"T1": "1", "T2": "2", "T3": "F"}


def binarize(survey):
    """Long format: one row per participant-timepoint with a boolean flag per domain."""
    rows = []
    for _, r in survey.iterrows():
        for tp, sfx in SUFFIX.items():
            rows.append({
                "deviceId": r["deviceId"], "timepoint": tp,
                "ISI": r[f"ISI_{sfx}"] >= CUTOFFS["ISI"],
                "PHQ9": r[f"PHQ9_{sfx}"] >= CUTOFFS["PHQ9"],
                "GAD7": r[f"GAD7_{sfx}"] >= CUTOFFS["GAD7"],
            })
    return pd.DataFrame(rows)


def combination_table(flags, unit_label):
    """Counts for all 8 combinations of the three binary flags."""
    combos = [
        ("None elevated", ~flags["ISI"] & ~flags["PHQ9"] & ~flags["GAD7"]),
        ("Insomnia only", flags["ISI"] & ~flags["PHQ9"] & ~flags["GAD7"]),
        ("Depression only", ~flags["ISI"] & flags["PHQ9"] & ~flags["GAD7"]),
        ("Anxiety only", ~flags["ISI"] & ~flags["PHQ9"] & flags["GAD7"]),
        ("Insomnia + Depression", flags["ISI"] & flags["PHQ9"] & ~flags["GAD7"]),
        ("Insomnia + Anxiety", flags["ISI"] & ~flags["PHQ9"] & flags["GAD7"]),
        ("Depression + Anxiety", ~flags["ISI"] & flags["PHQ9"] & flags["GAD7"]),
        ("All three", flags["ISI"] & flags["PHQ9"] & flags["GAD7"]),
    ]
    total = len(flags)
    return pd.DataFrame([
        {"Category": name, "n": int(mask.sum()),
         "Percent": f"{mask.sum() / total * 100:.1f}%", "Unit": unit_label}
        for name, mask in combos
    ])


def conditional_table(flags):
    """P(other domain elevated | this domain elevated) - the overlap that matters."""
    out = []
    for a in ("ISI", "PHQ9", "GAD7"):
        sub = flags[flags[a]]
        row = {"Given elevated": a, "n_with_condition": int(flags[a].sum())}
        for b in ("ISI", "PHQ9", "GAD7"):
            if a == b:
                continue
            pct = (sub[b].mean() * 100) if len(sub) else np.nan
            row[f"Also {b} (%)"] = f"{pct:.1f}%"
        out.append(row)
    return pd.DataFrame(out)


def jaccard_table(flags):
    """Set overlap between domains, independent of prevalence differences."""
    out = []
    for a, b in (("ISI", "PHQ9"), ("ISI", "GAD7"), ("PHQ9", "GAD7")):
        inter = int((flags[a] & flags[b]).sum())
        union = int((flags[a] | flags[b]).sum())
        out.append({"Pair": f"{a} & {b}", "Intersection": inter, "Union": union,
                    "Jaccard": f"{inter / union:.3f}" if union else "NA"})
    return pd.DataFrame(out)


def draw_venn(ax, counts, total, title):
    """Three-circle Venn drawn directly, since matplotlib_venn is not installed."""
    r = 0.55
    centres = {"ISI": (-0.30, 0.22), "PHQ9": (0.30, 0.22), "GAD7": (0.0, -0.30)}
    for k, c in centres.items():
        ax.add_patch(Circle(c, r, facecolor=COLORS[k], alpha=0.34,
                            edgecolor=COLORS[k], linewidth=2.0, zorder=2))

    # Region label positions matched to the circle geometry above.
    pos = {
        "Insomnia only": (-0.56, 0.50), "Depression only": (0.56, 0.50),
        "Anxiety only": (0.0, -0.70), "Insomnia + Depression": (0.0, 0.47),
        "Insomnia + Anxiety": (-0.40, -0.24), "Depression + Anxiety": (0.40, -0.24),
        "All three": (0.0, 0.02),
    }
    for name, (x, y) in pos.items():
        n = counts.get(name, 0)
        ax.text(x, y, f"{n}\n({n / total * 100:.0f}%)", ha="center", va="center",
                fontsize=11, fontweight="bold", zorder=3)

    ax.text(-0.86, 0.86, LABELS["ISI"], ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=COLORS["ISI"])
    ax.text(0.86, 0.86, LABELS["PHQ9"], ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=COLORS["PHQ9"])
    ax.text(0.0, -1.06, LABELS["GAD7"], ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=COLORS["GAD7"])

    # Placed top-left: the bottom of the panel is occupied by the GAD-7 label.
    none_n = counts.get("None elevated", 0)
    ax.text(-1.30, 1.02, f"No domain elevated: {none_n} ({none_n / total * 100:.0f}%)",
            ha="left", va="center", fontsize=9.5, style="italic", color="#444444")

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.22, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=12, pad=8)


def run_step7():
    print("=" * 50)
    print("STEP 7: Symptom Co-Occurrence (descriptive)")
    print("=" * 50)

    survey = load_survey_data()
    long = binarize(survey)
    n_subj = survey["deviceId"].nunique()

    # Participant level: elevated at any of the three assessments.
    ever = long.groupby("deviceId")[["ISI", "PHQ9", "GAD7"]].any().reset_index()

    tbl_ever = combination_table(ever, f"participants (n={n_subj}, ever elevated)")
    tbl_obs = combination_table(long, f"observations (n={len(long)}, pooled)")
    print(f"\nParticipant level (ever elevated across T1-T3), N = {n_subj}:")
    print(tbl_ever[["Category", "n", "Percent"]].to_string(index=False))
    print(f"\nObservation level (pooled across T1-T3), N = {len(long)}:")
    print(tbl_obs[["Category", "n", "Percent"]].to_string(index=False))

    combined = pd.concat([tbl_ever, tbl_obs], ignore_index=True)
    p6 = os.path.join(RESULTS_DIR, "Table6_Symptom_CoOccurrence.csv")
    combined.to_csv(p6, index=False, encoding="utf-8-sig")
    print(f"\nSaved Table 6 to: {p6}")

    cond = conditional_table(ever)
    jac = jaccard_table(ever)
    print("\nConditional co-occurrence (participant level, ever elevated):")
    print(cond.to_string(index=False))
    print("\nSet overlap (Jaccard):")
    print(jac.to_string(index=False))

    p6b = os.path.join(RESULTS_DIR, "Table6b_CoOccurrence_Conditional.csv")
    cond.to_csv(p6b, index=False, encoding="utf-8-sig")
    jac.to_csv(os.path.join(RESULTS_DIR, "Table6c_CoOccurrence_Jaccard.csv"),
               index=False, encoding="utf-8-sig")
    print(f"Saved Table 6b to: {p6b}")

    # How many domains are elevated per participant / per observation.
    ever["n_domains"] = ever[["ISI", "PHQ9", "GAD7"]].sum(axis=1)
    long["n_domains"] = long[["ISI", "PHQ9", "GAD7"]].sum(axis=1)
    dist_ever = ever["n_domains"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    print(f"\nDomains elevated per participant: {dist_ever.to_dict()}")

    counts_ever = dict(zip(tbl_ever["Category"], tbl_ever["n"]))
    counts_obs = dict(zip(tbl_obs["Category"], tbl_obs["n"]))

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(14, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.85], wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    draw_venn(ax1, counts_ever, n_subj,
              f"A. Participants ever elevated (N = {n_subj})")

    ax2 = fig.add_subplot(gs[0, 1])
    draw_venn(ax2, counts_obs, len(long),
              f"B. Pooled observations (N = {len(long)})")

    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar([0, 1, 2, 3], dist_ever.values,
                   color=["#b0b0b0", "#7fb3d5", "#e08a4a", "#8e6fb3"],
                   edgecolor="black", linewidth=0.8, alpha=0.9)
    for b, v in zip(bars, dist_ever.values):
        if v:
            ax3.text(b.get_x() + b.get_width() / 2, v + 0.4,
                     f"{v}\n({v / n_subj * 100:.0f}%)", ha="center",
                     fontsize=10, fontweight="bold")
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xlabel("Number of domains elevated")
    ax3.set_ylabel("Participants")
    ax3.set_title("C. Multi-domain burden", fontweight="bold", fontsize=12, pad=8)
    ax3.set_ylim(0, max(dist_ever.values) * 1.25)
    ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Figure 7. Symptom Co-Occurrence Across Insomnia, Depression, and Anxiety",
                 fontweight="bold", fontsize=14, y=0.99)
    fig.text(0.5, 0.015,
             "Cutoffs: ISI $\\geq$ 8, PHQ-9 $\\geq$ 5, GAD-7 $\\geq$ 5. "
             "Descriptive only; high co-occurrence is why overlapping SHAP rankings across "
             "outcomes indicate shared distress variance, not disorder specificity.",
             ha="center", fontsize=9, style="italic", color="#333333")
    plt.tight_layout(rect=[0, 0.045, 1, 0.96])

    p7 = os.path.join(RESULTS_DIR, "Figure7_Symptom_CoOccurrence.png")
    plt.savefig(p7, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 7 to: {p7}")

    # Consistency check against Table 1's pooled prevalences.
    print("\nCross-check vs Table 1 (pooled prevalence, should match):")
    for k, cut in CUTOFFS.items():
        print(f"  {k} >= {cut}: {long[k].mean() * 100:.1f}%")

    return combined, cond, jac


if __name__ == "__main__":
    run_step7()
