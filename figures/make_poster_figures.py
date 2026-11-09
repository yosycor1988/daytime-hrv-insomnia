"""Poster figures redrawn from the CSVs in ../results (no new computation).

Palette matches the poster: navy = wearable/HRV, teal = sleep diary, grey = activity.
Figures are drawn at their printed size on A0 so font sizes are the real point sizes.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUT_DIR = RESULTS  # figures are written next to the tables they are drawn from

NAVY = "#0B2A4A"
TEAL = "#0E9F8E"
TEAL_LIGHT = "#7FE3D2"
GREY = "#8796A6"
TEXT = "#1B1F24"
PANEL = "#EAF2F8"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 20,
    "axes.edgecolor": "#B8C4D0",
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "savefig.dpi": 300,
})

# Display order and short labels, most to least important for ISI.
FEATURES = [
    ("WASO Dev", "Δ WASO", TEAL),
    ("Sleep Efficiency Dev", "Δ Sleep efficiency", TEAL),
    ("Sleep Latency Dev", "Δ Sleep latency", TEAL),
    ("LF/HF (Daytime Dev)", "Δ LF/HF (HRV)", NAVY),
    ("SDNN (Daytime Dev)", "Δ SDNN (HRV)", NAVY),
    ("Heart Rate (Daytime Dev)", "Δ Heart rate", NAVY),
    ("Physical Activity (Steps Dev)", "Δ Steps", GREY),
]


def set_palette(diary, hrv, activity):
    """Re-colour every figure: diary/insomnia emphasis, wearable HRV, wearable activity."""
    global TEAL, NAVY, GREY, FEATURES
    TEAL, NAVY, GREY = diary, hrv, activity
    FEATURES = [(key, label, {0: diary, 1: hrv, 2: activity}[group])
                for (key, label, _), group in zip(FEATURES, (0, 0, 0, 1, 1, 1, 2))]


def fig_shap_bars(figsize=(10.0, 5.6), out="figP_shap_ranked.png"):
    """Mean |SHAP| for ISI (full sample) with bootstrap 95% interval."""
    shap = pd.read_csv(os.path.join(RESULTS, "Table3c_CrossDomain_SHAP_Matrix.csv"),
                       index_col=0, encoding="utf-8-sig")["Insomnia (ISI)"]
    boot = pd.read_csv(os.path.join(RESULTS, "Table_Bootstrap_Rank_Stability.csv"),
                       encoding="utf-8-sig").set_index("Feature")

    fig, ax = plt.subplots(figsize=figsize)
    for i, (key, label, color) in enumerate(FEATURES):
        val = shap[key]
        lo, hi = boot.loc[key, "SHAP_2.5%"], boot.loc[key, "SHAP_97.5%"]
        ax.barh(i, val, color=color, height=0.68, zorder=2)
        ax.plot([lo, hi], [i, i], color="#5A6878", lw=2, zorder=3)
        ax.plot([lo, lo], [i - 0.15, i + 0.15], color="#5A6878", lw=2, zorder=3)
        ax.plot([hi, hi], [i - 0.15, i + 0.15], color="#5A6878", lw=2, zorder=3)
        ax.text(hi + 0.04, i, f"{val:.2f}", va="center", fontsize=18, color=TEXT)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels([f[1] for f in FEATURES])
    ax.invert_yaxis()
    ax.set_xlim(0, 2.6)
    ax.set_xlabel("Mean |SHAP| (ISI points)   ·   whiskers: bootstrap 95% interval")

    # Bracket the three sleep-diary rows: the take-home of this figure.
    bx = 2.03
    ax.plot([bx, bx + 0.05, bx + 0.05, bx], [-0.3, -0.3, 2.3, 2.3], color=TEAL, lw=3, clip_on=False)
    ax.text(bx + 0.11, 1.0, "Top 3:\nsleep\ndiary", va="center", ha="left", fontsize=19,
            color=TEAL, fontweight="bold", linespacing=1.05)
    fig.text(0.01, 0.01, "Δ = deviation from the participant's own 14-day baseline",
             fontsize=15, color="#5A6878", style="italic")
    ax.xaxis.grid(True, color="#E3E9EF", zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (TEAL, NAVY, GREY)]
    ax.legend(handles, ["Sleep diary", "Wearable HRV", "Wearable activity"],
              loc="lower right", frameon=False, fontsize=18)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT_DIR, out))
    plt.close(fig)


def fig_crossdomain_heatmap():
    """Mean |SHAP| by outcome, with each model's held-out R² in the column header."""
    shap = pd.read_csv(os.path.join(RESULTS, "Table3c_CrossDomain_SHAP_Matrix.csv"),
                       index_col=0, encoding="utf-8-sig")
    perf = pd.read_csv(os.path.join(RESULTS, "Table3_Cross_Domain_Overlap_Performance.csv"),
                       encoding="utf-8-sig").set_index("Outcome")

    cols = list(shap.columns)
    short = {"Insomnia (ISI)": "Insomnia\n(ISI)", "Depression (PHQ-9)": "Depression\n(PHQ-9)",
             "Anxiety (GAD-7)": "Anxiety\n(GAD-7)", "Distress Composite": "Distress\ncomposite"}
    rows = [f[0] for f in FEATURES]
    data = shap.loc[rows, cols].values

    cmap = LinearSegmentedColormap.from_list("poster", ["#FFFFFF", TEAL_LIGHT, TEAL, NAVY])
    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    ax.imshow(data, cmap=cmap, vmin=0, vmax=0.85, aspect="auto")
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=19,
                    color="white" if v > 0.45 else TEXT,
                    fontweight="bold" if c == 0 and r < 3 else "normal")
    labels = []
    for col in cols:
        r2, p = perf.loc[col, "LOSO_R2"], perf.loc[col, "Permutation_p"]
        r2_txt = f"{r2:+.2f}".replace("-", "−")
        labels.append(f"{short[col]}\nR² {r2_txt}\np = {p:.3f}")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=17)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f[1] for f in FEATURES])
    for t, (_, _, color) in zip(ax.get_yticklabels(), FEATURES):
        t.set_color(color if color != GREY else "#5A6878")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color="white", lw=3)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figP_crossdomain_heatmap.png"))
    plt.close(fig)


def fig_cooccurrence_venn():
    """Three-set Venn of participants ever above cutoff, from Table 6 counts."""
    t = pd.read_csv(os.path.join(RESULTS, "Table6_Symptom_CoOccurrence.csv"), encoding="utf-8-sig")
    t = t[t["Unit"].str.startswith("participants")].set_index("Category")["n"]
    n_total = int(t.sum())

    fig, ax = plt.subplots(figsize=(10.3, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")
    r = 1.0
    centers = {"I": (-0.55, 0.32), "D": (0.55, 0.32), "A": (0.0, -0.58)}
    colors = {"I": TEAL, "D": NAVY, "A": GREY}
    for k, (x, y) in centers.items():
        ax.add_patch(Circle((x, y), r, facecolor=colors[k], alpha=0.22, edgecolor=colors[k], lw=2.5))

    def lab(x, y, key, bold=False):
        n = int(t[key])
        ax.text(x, y, f"{n}\n({100 * n / n_total:.0f}%)", ha="center", va="center",
                fontsize=23, color=TEXT, fontweight="bold" if bold else "normal", linespacing=0.95)

    lab(-1.02, 0.55, "Insomnia only")
    lab(1.02, 0.55, "Depression only")
    lab(0.0, -1.18, "Anxiety only")
    lab(0.0, 0.78, "Insomnia + Depression")
    lab(-0.62, -0.42, "Insomnia + Anxiety")
    lab(0.62, -0.42, "Depression + Anxiety")
    lab(0.0, 0.02, "All three", bold=True)

    ax.text(-1.25, 1.48, "Insomnia\nISI ≥ 8", ha="center", va="bottom", fontsize=19,
            color=TEAL, fontweight="bold")
    ax.text(1.25, 1.48, "Depression\nPHQ-9 ≥ 5", ha="center", va="bottom", fontsize=19,
            color=NAVY, fontweight="bold")
    ax.text(1.25, -1.45, "Anxiety\nGAD-7 ≥ 5", ha="left", va="center", fontsize=19,
            color="#5A6878", fontweight="bold")
    none = int(t["None elevated"])
    ax.text(-2.85, -1.45, f"None elevated:\n{none} ({100 * none / n_total:.0f}%)", ha="left",
            va="center", fontsize=18, color="#5A6878", style="italic")
    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-1.75, 2.05)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT_DIR, "figP_cooccurrence_venn.png"))
    plt.close(fig)


def fig_permutation_crop(line_color=TEAL, out="figP_permutation_null.png"):
    """Right panel (permutation null) of the pipeline's Figure 2, cropped.

    The null distribution itself was not saved, so the panel is reused rather than redrawn;
    only the red observed-R² line is recoloured to the poster teal (pixels, not data).
    """
    import numpy as np

    im = Image.open(os.path.join(RESULTS, "Figure2_ISI_Model_Validation.png")).convert("RGB")
    w, h = im.size
    im = im.crop((int(w * 0.505), 0, w, h))
    a = np.asarray(im).astype(float)
    redness = np.clip((a[..., 0] - np.maximum(a[..., 1], a[..., 2])) / 255.0, 0, 1)
    weight = np.clip(redness / 0.8, 0, 1)[..., None]
    teal = np.array([int(line_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    white = np.array([255.0, 255.0, 255.0])
    target = white + (teal - white) * weight  # teal at full strength, fading with anti-aliasing
    a = np.where(weight > 0.05, target, a)
    Image.fromarray(a.astype("uint8")).save(os.path.join(OUT_DIR, out), dpi=(300, 300))


def _person(ax, x, y, r, face, figure):
    """Round badge with a simple person glyph (head + shoulders)."""
    from matplotlib.patches import Circle, FancyBboxPatch

    ax.add_patch(Circle((x, y), r, color=face, zorder=2))
    ax.add_patch(Circle((x, y + 0.32 * r), 0.24 * r, color=figure, zorder=3))
    ax.add_patch(FancyBboxPatch((x - 0.36 * r, y - 0.62 * r), 0.72 * r, 0.62 * r,
                                boxstyle=f"round,pad=0,rounding_size={0.3 * r}", color=figure, zorder=3))


def fig_sample_icons(figsize=(13.6, 4.6), out="figP_icons_sample.png"):
    """Sample-and-design infographic: who, how many observations, which features."""
    items = [
        ("49 ADULTS", "Healthy volunteers, 28.3 ± 5.9 y,\n51% female, four weeks\nof Galaxy Watch Active 2", NAVY, "white"),
        ("98 OBSERVATIONS", "Two follow-up assessments\nper person (baseline visit\nhas no prior data)", TEAL, "white"),
        ("7 FEATURES", "Deviations from each person's\n14-day baseline: 3 HRV,\n3 sleep-diary, 1 activity", TEAL_LIGHT, NAVY),
    ]
    fig, ax = plt.subplots(figsize=figsize)
    y0, y1 = 0.12, 1.55
    width = (y1 - y0) * figsize[0] / figsize[1]  # equal aspect so the badges stay round
    ax.set_xlim(0, width)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    xs = [width / 6, width / 2, 5 * width / 6]
    ax.plot([xs[1], xs[1]], [1.33, 1.17], color="#C9D6E3", lw=3, zorder=1)
    ax.plot([xs[0], xs[2]], [1.17, 1.17], color="#C9D6E3", lw=3, zorder=1)
    for x in xs:
        ax.plot([x, x], [1.17, 1.02], color="#C9D6E3", lw=3, zorder=1)
    _person(ax, xs[1], 1.42, 0.11, NAVY, "white")
    for x, (head, body, face, figure) in zip(xs, items):
        _person(ax, x, 0.92, 0.1, face, figure)
        ax.text(x, 0.72, head, ha="center", va="center", fontsize=26, fontweight="bold", color=NAVY)
        ax.text(x, 0.58, body, ha="center", va="top", fontsize=19, color="#4A5866", linespacing=1.25)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(OUT_DIR, out), transparent=True)  # sits on a tinted panel
    plt.close(fig)


def fig_variance_circles(figsize=(6.0, 5.6), out="figP_variance_circles.png"):
    """Nested circles: total ISI variance, the between-person part, and what LOSO can use."""
    from matplotlib.patches import Circle

    t2 = pd.read_csv(os.path.join(RESULTS, "Table2_ISI_Model_Performance.csv"),
                     encoding="utf-8-sig").set_index("Metric")["Value"]
    subj = float(str(t2["Benchmark: subject-mean predictor RMSE"]).split()[0])
    r2 = float(str(t2["LOSO-CV R²"]).split()[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.0, color=NAVY))
    ax.add_patch(Circle((0, -0.2), 0.78, color=TEAL))
    ax.add_patch(Circle((0, -0.43), 0.53, color=TEAL_LIGHT))
    ax.text(0, 0.82, "TOTAL ISI VARIANCE", ha="center", va="center", fontsize=19, fontweight="bold", color="white")
    ax.text(0, 0.42, "BETWEEN-PERSON", ha="center", va="center", fontsize=19, fontweight="bold", color="white")
    ax.text(0, 0.29, f"subject-mean RMSE {subj:.2f}", ha="center", va="center", fontsize=16, color="white")
    ax.text(0, -0.27, "WITHIN-PERSON", ha="center", va="center", fontsize=19, fontweight="bold", color=NAVY)
    ax.text(0, -0.42, "what LOSO can use", ha="center", va="center", fontsize=16, color=NAVY)
    ax.text(0, -0.6, f"R² = {r2:.3f}", ha="center", va="center", fontsize=22, fontweight="bold", color=NAVY)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(OUT_DIR, out), transparent=True)
    plt.close(fig)


def fig_model_vs_null(figsize=(9.0, 6.0), out="figP_model_vs_null.png"):
    """Held-out RMSE of the insomnia model against the two null benchmarks."""
    t2 = pd.read_csv(os.path.join(RESULTS, "Table2_ISI_Model_Performance.csv"),
                     encoding="utf-8-sig").set_index("Metric")["Value"]
    t5 = pd.read_csv(os.path.join(RESULTS, "Table5_Imputation_Sensitivity.csv"), encoding="utf-8-sig")
    rmse = lambda k: float(str(t2[k]).split()[0])
    bars = [
        ("Subject-mean\nnull", rmse("Benchmark: subject-mean predictor RMSE"), NAVY),
        ("Model\n(complete cases)", float(t5.loc[t5["Sample"].str.startswith("HRV"), "LOSO_RMSE"].iloc[0]), TEAL),
        ("Model\n(primary)", rmse("LOSO-CV RMSE"), TEAL_LIGHT),
        ("Grand-mean\nnull", rmse("Benchmark: grand-mean predictor RMSE"), "#C9D6E3"),
    ]
    sd = rmse("Outcome SD (ISI)")
    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, v, c) in enumerate(bars):
        ax.bar(i, v, color=c, width=0.62, zorder=2)
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=20, fontweight="bold", color=TEXT)
    ax.axhline(sd, color="#5A6878", lw=2, ls="--", zorder=1)
    ax.text(-0.3, sd + 0.08, f"ISI SD = {sd:.2f}", ha="left", va="bottom", fontsize=16, color="#5A6878")
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=17)
    ax.set_ylabel("RMSE (ISI points) · lower is better", fontsize=17)
    ax.set_ylim(0, 5.4)
    ax.yaxis.grid(True, color="#E3E9EF", zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out))
    plt.close(fig)


def fig_outcome_specificity(figsize=(9.0, 6.0), out="figP_outcome_specificity.png"):
    """Held-out R² for both insomnia models and the three other outcomes, zero line marked."""
    perf = pd.read_csv(os.path.join(RESULTS, "Table3_Cross_Domain_Overlap_Performance.csv"),
                       encoding="utf-8-sig").set_index("Outcome")
    t5 = pd.read_csv(os.path.join(RESULTS, "Table5_Imputation_Sensitivity.csv"), encoding="utf-8-sig")
    cc = t5[t5["Sample"].str.startswith("HRV")].iloc[0]
    bars = [("Insomnia\ncomplete cases", float(cc["LOSO_R2"]), float(cc["Permutation_p"]))]
    for key, label in (("Insomnia (ISI)", "Insomnia\nprimary"), ("Distress Composite", "Distress\ncomposite"),
                       ("Depression (PHQ-9)", "Depression\nPHQ-9"), ("Anxiety (GAD-7)", "Anxiety\nGAD-7")):
        bars.append((label, perf.loc[key, "LOSO_R2"], perf.loc[key, "Permutation_p"]))
    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, r2, p) in enumerate(bars):
        good = r2 > 0 and p < 0.05
        ax.bar(i, r2, color=TEAL if good else "#C9D6E3", width=0.62, zorder=2)
        y = r2 + 0.012 if r2 > 0 else r2 - 0.012
        ax.text(i, y, f"p = {p:.3f}", ha="center", va="bottom" if r2 > 0 else "top", fontsize=16,
                color=TEAL if good else "#5A6878", fontweight="bold" if good else "normal")
    ax.axhline(0, color=TEXT, lw=2, zorder=3)
    ax.text(0.5, 0.215, "better than the mean", ha="center", fontsize=17, color=TEAL, fontweight="bold")
    ax.text(3.0, -0.335, "no signal", ha="center", fontsize=17, color="#5A6878", fontweight="bold")
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=16)
    ax.set_ylabel("Held-out R² (LOSO-CV)", fontsize=17)
    ax.set_ylim(-0.36, 0.25)
    ax.yaxis.grid(True, color="#E3E9EF", zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out))
    plt.close(fig)


def fig_pipeline_trim(src="h_fig1_pipeline.png", out="figP_pipeline_h.png", pad=20):
    """Trim the white margin around the horizontal pipeline diagram (a hand-drawn image)."""
    from PIL import ImageOps

    path = os.path.join(HERE, src)
    if not os.path.exists(path):  # the diagram is not part of the public repository
        print(f"Skipping pipeline diagram: {src} not found")
        return
    im = Image.open(path).convert("RGB")
    left, top, right, bottom = ImageOps.invert(im.convert("L")).point(lambda v: 255 if v > 12 else 0).getbbox()
    w, h = im.size
    im.crop((max(left - pad, 0), max(top - pad, 0), min(right + pad, w), min(bottom + pad, h))).save(
        os.path.join(OUT_DIR, out), dpi=(300, 300))


def fig_symptom_prevalence(figsize=(10.3, 5.2), out="figP_symptom_prevalence.png"):
    """Participants ever above each cutoff, plus how many domains each participant had."""
    t = pd.read_csv(os.path.join(RESULTS, "Table6_Symptom_CoOccurrence.csv"), encoding="utf-8-sig")
    t = t[t["Unit"].str.startswith("participants")].set_index("Category")["n"]
    n = int(t.sum())
    ever = {
        "Insomnia (ISI ≥ 8)": t["Insomnia only"] + t["Insomnia + Depression"] + t["Insomnia + Anxiety"] + t["All three"],
        "Depression (PHQ-9 ≥ 5)": t["Depression only"] + t["Insomnia + Depression"] + t["Depression + Anxiety"] + t["All three"],
        "Anxiety (GAD-7 ≥ 5)": t["Anxiety only"] + t["Insomnia + Anxiety"] + t["Depression + Anxiety"] + t["All three"],
    }
    domains = [
        t["None elevated"],
        t["Insomnia only"] + t["Depression only"] + t["Anxiety only"],
        t["Insomnia + Depression"] + t["Insomnia + Anxiety"] + t["Depression + Anxiety"],
        t["All three"],
    ]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1.35]})
    colors = [TEAL, NAVY, GREY]
    for i, ((label, k), c) in enumerate(zip(ever.items(), colors)):
        pct = 100 * k / n
        ax.barh(i, 100, color="#EEF2F6", height=0.62)
        ax.barh(i, pct, color=c, height=0.62)
        ax.text(pct + 1.5, i, f"{k}/{n} ({pct:.0f}%)", va="center", fontsize=20, color=TEXT,
                fontweight="bold")
    ax.set_yticks(range(3))
    ax.set_yticklabels(list(ever.keys()), fontsize=20)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    # One stacked bar: number of domains elevated per participant.
    shades = ["#D5DEE7", TEAL_LIGHT, TEAL, NAVY]
    left = 0
    for k_dom, (cnt, col) in enumerate(zip(domains, shades)):
        pct = 100 * cnt / n
        ax2.barh(0, pct, left=left, color=col, height=0.75, edgecolor="white", lw=2)
        ax2.text(left + pct / 2, 0, f"{k_dom}: {cnt} ({pct:.0f}%)", ha="center", va="center",
                 fontsize=17, color="white" if k_dom >= 2 else TEXT, fontweight="bold")
        left += pct
    ax2.set_xlim(0, 100)
    ax2.set_yticks([0])
    ax2.set_yticklabels(["Domains\nelevated"], fontsize=19)
    ax2.set_xticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.tick_params(length=0)
    multi = domains[2] + domains[3]
    ax2.set_title(f"{multi}/{n} ({100 * multi / n:.0f}%) were above cutoff on two or more domains",
                  fontsize=18, color="#5A6878", style="italic", loc="right", pad=8)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(os.path.join(OUT_DIR, out))
    plt.close(fig)


def fig_outcome_r2(figsize=(10.0, 4.3), out="figP_outcome_r2.png"):
    """Held-out (LOSO-CV) R² for each outcome, with a reference line at zero."""
    perf = pd.read_csv(os.path.join(RESULTS, "Table3_Cross_Domain_Overlap_Performance.csv"),
                       encoding="utf-8-sig").set_index("Outcome")
    names = {"Insomnia (ISI)": "Insomnia (ISI)", "Depression (PHQ-9)": "Depression (PHQ-9)",
             "Anxiety (GAD-7)": "Anxiety (GAD-7)", "Distress Composite": "Distress composite"}

    fig, ax = plt.subplots(figsize=figsize)
    for i, (key, label) in enumerate(names.items()):
        r2, p = perf.loc[key, "LOSO_R2"], perf.loc[key, "Permutation_p"]
        color = TEAL if r2 > 0 and p < 0.05 else GREY
        ax.barh(i, r2, color=color, height=0.62, zorder=2)
        txt = f"{r2:+.3f}   p = {p:.3f}".replace("-", "−")
        if r2 > 0:
            ax.text(r2 + 0.01, i, txt, va="center", ha="left", fontsize=19, color=TEAL, fontweight="bold")
        else:
            ax.text(0.01, i, txt, va="center", ha="left", fontsize=19, color="#5A6878")
    ax.axvline(0, color=TEXT, lw=2, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(names.values()), fontsize=20)
    ax.invert_yaxis()
    ax.set_xlim(-0.30, 0.30)
    ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    ax.set_xticklabels(["−0.3", "−0.2", "−0.1", "0", "0.1", "0.2", "0.3"])
    ax.set_xlabel("Held-out R² (LOSO-CV)")
    ax.xaxis.grid(True, color="#E3E9EF", zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.01, 0.01, "R² < 0: less accurate than predicting the sample mean (not a negative association)",
             fontsize=15, color="#5A6878", style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT_DIR, out))
    plt.close(fig)


if __name__ == "__main__":
    fig_shap_bars()
    fig_crossdomain_heatmap()
    fig_cooccurrence_venn()
    fig_permutation_crop()
    fig_symptom_prevalence()
    fig_outcome_r2()
    fig_pipeline_trim()
    # Landscape layout (v6): SHAP drawn wide and short to fit its slot.
    fig_shap_bars(figsize=(13.6, 5.0), out="figP_shap_wide.png")
    # Infographic layout (v7): the design's figures and infographics redrawn in our palette
    # with poster-sized type.
    fig_model_vs_null()
    fig_outcome_specificity()
    fig_sample_icons()
    fig_variance_circles()

    print("Poster figures written to", os.path.abspath(OUT_DIR))
