# Comparative Evaluation of Diffusion, GAN, and VAE Models for Synthetic Clinical Tabular Data Generation

**Focus:** Statistical fidelity, downstream predictive utility, and privacy-risk indicators for synthetic hospital-encounter data.

This capstone compares four saved experiments from three generative model families: **TVAE (50 epochs)**, **TVAE (300 epochs)**, **CTGAN (300 epochs)**, and a **TabDDPM-inspired joint Gaussian–multinomial diffusion model (100 epochs)**. The study uses the [UCI Diabetes 130-US Hospitals dataset (1999–2008), dataset ID 296](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

> **Research only.** The generated datasets and models are not validated for clinical decision-making, are not certified anonymous, and should not be shared as privacy-safe data without additional review.

## Research questions

1. Which model best preserves numerical distributions, categorical distributions, inter-feature correlations, and the relatively rare 30-day readmission outcome?
2. How well does a classifier trained on synthetic data generalize to **held-out real patients** (train-on-synthetic, test-on-real; TSTR)?
3. What do sampled nearest-neighbor and exact-row-match tests reveal about possible memorization or membership-discrimination risk?

## Dataset and preprocessing

- Original source: **101,766 hospital encounters**.
- Excluded: **2,423** death/hospice-coded encounters and **3** records with invalid gender.
- Eligible cohort: **99,340 encounters**.
- Patient-grouped split (`GroupShuffleSplit`, seed 42): **79,567 train encounters** / **19,773 held-out test encounters** with **zero overlapping patients**.
- Model table: **45 columns** — 8 numerical predictors, 36 categorical predictors, and one binary target, `readmitted_30`.
- Target: **1** for readmission within 30 days (`<30`), **0** otherwise (`>30` or `NO`).
- Removed patient/encounter identifiers from model inputs. Grouped diagnosis codes by ICD-9-CM families, preserved missing categories explicitly, and learned rare-specialty groupings and numerical imputation values from the training split only.

The held-out real test split was reserved for downstream utility evaluation, not generator fitting. See `src/preprocessing.py` and `notebooks/02_preprocessing.ipynb`.

## Experiments

| Generator | Training setting | Saved synthetic sample size |
|---|---|---:|
| TVAE baseline | 50 epochs | 79,567 |
| TVAE extended | 300 epochs | 79,567 |
| CTGAN | 300 epochs | 79,567 |
| TabDDPM-inspired mixed diffusion | 100 epochs | 79,567 |

**Implementation note:** The diffusion experiment is *TabDDPM-inspired*, not an exact reproduction of the reference TabDDPM implementation. CTGAN training was completed through `src/ctgan_train_background.py` after notebook/SSH interruptions; its notebook's execution history therefore differs from the saved training log.

## Results

### Statistical fidelity and predictive utility

| Model | Synthetic readmission rate | Mean numerical KS ↓ | Mean categorical TVD ↓ | Real-category coverage ↑ | TSTR AUROC ↑ | TSTR AUPRC ↑ |
|---|---:|---:|---:|---:|---:|---:|
| TVAE 50 | 0.000% | 0.1805 | 0.0902 | 53.65% | 0.5000* | 0.1135* |
| TVAE 300 | 0.043% | 0.1387 | 0.0720 | 66.67% | 0.6007 | 0.1735 |
| CTGAN 300 | 4.121% | 0.1720 | 0.0846 | **100.00%** | 0.4966 | 0.1128 |
| **TabDDPM 100** | **10.566%** | **0.0378** | **0.0143** | 91.93% | **0.6467** | **0.2095** |

The **real training** readmission rate is **11.399%**. The real-trained/real-tested (**TRTR**) logistic-regression baseline on held-out patients achieves **AUROC 0.6709** and **AUPRC 0.2264**.

*TVAE 50 generated no positive readmission rows. Its TSTR scores come from a single-class fallback, **not** from a fitted two-class logistic regression.*

Within the tested models, **TabDDPM 100** performed best on measured statistical fidelity and TSTR discrimination. **CTGAN 300** achieved the highest categorical coverage. These results should not be interpreted as an equal-compute benchmark: model families and epoch budgets differ.

### Privacy-risk indicators

- **Zero exact full-row matches** to real training records were found in the evaluated synthetic tables.
- Sampled proximity-based membership **proxy AUROC** values were approximately **0.49–0.50**, using 1,000 real training queries, 1,000 held-out real queries, and a pool of 5,000 synthetic records per generator.
- These are **limited heuristic checks**. They do not prove anonymity, differential privacy, or safety to release individual synthetic records or model checkpoints.

## Repository organization

```text
notebooks/
  01_dataset_eda.ipynb
  02_preprocessing.ipynb
  03_tvae_training.ipynb
  04_tvae_diagnostics.ipynb
  05_tvae_300_epochs.ipynb
  06_ctgan_training.ipynb
  07_tabddpm_training.ipynb
  08_fidelity_evaluation.ipynb
  09_utility_tstr.ipynb
  10_privacy_risk_evaluation.ipynb
  11_final_comparison.ipynb
src/
  preprocessing.py
  ctgan_train_background.py
```

Local output folders including `data/`, `models/`, and `results/` are intentionally excluded from version control by `.gitignore`. This repository publishes methods and documented aggregate findings, **not** patient-level tables, generated rows, raw identifier manifests, or model weights. A small, curated set of aggregate tables and figures may be published separately after review.

## How to explore or reproduce

1. Read the numbered notebooks in order and consult `src/preprocessing.py` for the patient-disjoint split and semantic feature treatment.
2. Fetch the original data from [UCI dataset 296](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) rather than using any published patient-level files from this repository.
3. Create a suitable Python environment and install the dependencies used by each notebook. **An environment/requirements specification is being prepared.** Some notebook cells were executed in Google Colab; training was also conducted on a CUDA-enabled Linux GPU server. Environment-specific paths and mounts must be adapted before running locally.
4. To reproduce the saved comparison, first regenerate the intermediate fidelity, utility, and privacy result files from notebooks 08–10; notebook 11 aggregates those outputs. Training the models from scratch requires substantial GPU time.

## Scope and limitations

- One retrospective source dataset and one patient-disjoint train/test split; external validation and confidence intervals were not established.
- Unequal epochs and computational budgets across generator families.
- TSTR used a single downstream logistic-regression evaluation pipeline; F1 and recall depend on the chosen threshold.
- Statistical similarity does not guarantee clinical consistency or fairness across demographic subgroups.
- Membership results are sampled proximity indicators, not comprehensive privacy attacks or a release certification.
- Reproducibility currently depends on environment-specific training scripts and notebook setup. Exact package versions and standardized execution instructions should be added before treating the repository as fully reproducible.

## Dataset attribution

UCI Machine Learning Repository: **Diabetes 130-US Hospitals for Years 1999–2008** (dataset ID 296). See the dataset page for the original publication, license, and required attribution. This repository does not redistribute the source records.
