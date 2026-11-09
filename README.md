# Interpretable Digital Phenotyping of Subclinical Insomnia From Real-World Wearable Daytime HRV and Sleep Diary Data

**Yosanly Cornelio, MD, MSc, PhD Candidate** · **Ismael Villanueva-Miranda, PhD**
University of Texas Southwestern Medical Center, Dallas, Texas

Poster presented at the Neuroscience of the Everyday World (N.E.W.) Conference, Boston University, November 12–13, 2026.

📄 [Abstract (PDF)](paper/Abstract_NEW2026_Cornelio.pdf) · ✉️ yosanly.corneliopuello@utsouthwestern.edu

---

## The finding, in one sentence

A modest but significant signal of subclinical insomnia in real-world data, carried mostly by the sleep diary rather than by daytime heart rate variability (HRV).

## Abstract

**Background.** Wearable daytime HRV is seldom examined for subclinical mental health, and pipelines are rarely reproducible end-to-end.

**Methods.** We analysed a public dataset of 49 healthy adults monitored for four weeks with a Samsung Galaxy Watch Active 2, with ISI, PHQ-9 and GAD-7 at three timepoints (Baigutanova et al., 2025). The watch was removed overnight, so all HRV is daytime and all sleep measures are self-reported. Timestamps were converted from UTC to Korea local time (KST, UTC+9) and analyses were restricted to 08:00–23:00 KST. Five-minute SDNN, LF/HF and heart rate were residualized on concurrent steps and local hour of day, aggregated to daily medians and combined with sleep-diary latency, wake after sleep onset (WASO) and efficiency. Each feature was expressed as a deviation from the participant's own rolling baseline (14-day primary, 7-day sensitivity), giving 98 observations from 49 participants. Seven deviation features entered a Random Forest evaluated by leave-one-subject-out cross-validation (LOSO-CV), with 1,000 label permutations, 1,000 subject-level bootstraps, TreeSHAP attributions, a linear mixed model, and grand-mean and subject-mean null benchmarks.

**Results.** Insomnia severity was predicted in held-out participants (LOSO-CV R² = 0.068, RMSE 4.39 vs 4.55 for a grand-mean predictor; permutation p = 0.004). Excluding zero-imputed observations strengthened the result (R² = 0.142, p = 0.001). The effect was modest: a subject-mean predictor reached RMSE 1.29, so stable between-person differences dominate. Sleep-diary features ranked 1–3 by SHAP; daytime HRV ranked 4–6 and steps last. Depression, anxiety and a distress composite were not predicted (all R² < 0). A 14-day baseline was required (7-day R² = −0.111).

**Conclusion.** Individualized deviation features predicted subclinical insomnia in unseen participants, but the ISI is itself a self-reported sleep instrument, so much of the signal plausibly reflects shared method variance. The contribution is a reproducible pipeline with null-model benchmarks and a quantified ceiling on what daytime HRV adds beyond a sleep diary.

## Repository contents

```
├── README.md
├── paper/             Conference abstract (PDF)
├── src/               Analysis pipeline (Python)
├── figures/           Script that draws the poster figures from the pipeline's tables
└── requirements.txt
```

Tables and figures are not included: running the pipeline writes them to `results/`.

## Data

This repository does **not** redistribute the data. Download the original dataset from its source and place it in `data/` before running the pipeline:

- Dataset: https://doi.org/10.6084/m9.figshare.28509740
- License: CC-BY 4.0

## Reproduce

```bash
pip install -r requirements.txt
python src/verify_timezone.py   # optional: checks the UTC -> KST correction
python src/run_pipeline.py      # features, LOSO-CV, permutations, SHAP, tables in results/
python figures/make_poster_figures.py   # poster figures, also in results/
```

## References
1. Wang P, Liu H, Shi Y, Liu A, Zhu Q, Albu I, et al. Med Research. 2025;1(2):226–238. doi:10.1002/mdr2.70017
2. Fechtelpeter J, Rauschenberg C, Goetzl C, Hiller S, Wierzba E, Emonds N, et al. npj Digit Med. 2026;9(1):70. doi:10.1038/s41746-025-02252-3
3. Baigutanova A, Park S, Constantinides M, Lee SW, Quercia D, Cha M. Sci Data. 2025;12(1):1474. doi:10.1038/s41597-025-05801-3
4. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. Heart rate variability: standards of measurement, physiological interpretation and clinical use. Circulation. 1996;93(5):1043–65. doi:10.1161/01.CIR.93.5.1043
5. Lundberg SM, Erion G, Chen H, DeGrave A, Prutkin JM, Nair B, et al. From local explanations to global understanding with explainable AI for trees. Nat Mach Intell. 2020;2(1):56–67. doi:10.1038/s42256-019-0138-9
6. Morin CM, Jarrin DC, Ivers H, Mérette C, LeBlanc M, Savard J. Incidence, persistence, and remission rates of insomnia over 5 years. JAMA Netw Open. 2020;3(11):e2018782. doi:10.1001/jamanetworkopen.2020.18782
7. Bastien CH, Vallières A, Morin CM. Validation of the Insomnia Severity Index as an outcome measure for insomnia research. Sleep Med. 2001;2(4):297–307. doi:10.1016/S1389-9457(00)00065-4
8. Kroenke K, Spitzer RL, Williams JB. The PHQ-9: validity of a brief depression severity measure. J Gen Intern Med. 2001;16(9):606–13. doi:10.1046/j.1525-1497.2001.016009606.x
9. Spitzer RL, Kroenke K, Williams JB, Löwe B. A brief measure for assessing generalized anxiety disorder: the GAD-7. Arch Intern Med. 2006;166(10):1092–7. doi:10.1001/archinte.166.10.1092
10. Saeb S, Lonini L, Jayaraman A, Mohr DC, Kording KP. The need to approximate the use-case in clinical machine learning. GigaScience. 2017;6(5):gix019. doi:10.1093/gigascience/gix019
11. Dodds KL, Miller CB, Kyle SD, Marshall NS, Gordon CJ. Heart rate variability in insomnia patients: a critical review of the literature. Sleep Med Rev. 2017;33:88–100. doi:10.1016/j.smrv.2016.06.004

## How to cite

Cornelio Y, Villanueva-Miranda I. Interpretable digital phenotyping of subclinical insomnia from real-world wearable daytime HRV and sleep diary data [poster]. Neuroscience of the Everyday World Conference; 2026 Nov 12–13; Boston, MA.
