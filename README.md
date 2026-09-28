# Beyond Static Rules: Machine Learning for Content Decay Prioritization in Search Operations

[![Live Report](https://img.shields.io/badge/Live%20Report-GitHub%20Pages-2ea44f?style=for-the-badge&logo=github)](https://nashidulsarker.github.io/flyrank-internship-capstone/)

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Validation: GroupKFold](https://img.shields.io/badge/Validation-5--Fold%20GroupKFold%20(Client)--blueviolet.svg)]()
[![Model: Dual--Champion](https://img.shields.io/badge/Dual--Champion-Random%20Forest%20%2B%20LightGBM-emerald.svg)]()

> 🌐 **Interactive Web Report:** [nashidulsarker.github.io/flyrank-internship-capstone](https://nashidulsarker.github.io/flyrank-internship-capstone/)

This repository contains the complete research pipeline, data contracts, signal audits, dual-model benchmarks, leakage stress tests, and editorial decision playbooks for prioritizing organic search content decay across enterprise publishing networks.

---

## Quick Start: The Executive Narrative

To experience the complete end-to-end research narrative in a single, fluid run:

Open and run **[`notebooks/capstone_content_decay.ipynb`](notebooks/capstone_content_decay.ipynb)**:
- **Duration:** ~35 seconds
- **Contents:** Problem framing $\rightarrow$ Data contract assertions $\rightarrow$ Audited SEO dogmas with effect sizes $\rightarrow$ Single & composite heuristic baselines $\rightarrow$ 5-Fold Client-Grouped CV $\rightarrow$ Dual-Model benchmarking $\rightarrow$ Controlled confession test & Memorization Gap $\rightarrow$ Editorial Playbook with client diversification quotas $\rightarrow$ Inline publication figures.

---

## Deliverables & Documentation

- **Interactive Web Report:** [Live Website on GitHub Pages](https://nashidulsarker.github.io/flyrank-internship-capstone/) ([`docs/index.html`](docs/index.html))
- **Research Paper:** [`docs/research_paper.md`](docs/research_paper.md) (All 9 canonical sections)
- **Master Capstone Notebook:** [`notebooks/capstone_content_decay.ipynb`](notebooks/capstone_content_decay.ipynb)
- **Automated Pipeline Runner:** `python scripts/run_all.py`

---

## Key Empirical Findings

1. **Dual-Model Operational Hierarchy:**
   - **Micro-Sprint Champion (Random Forest):** Achieves **100.00% Precision@20** and **88.00% Precision@50** (+22.00% absolute lift over production rule baselines), delivering unmatched reliability for weekly editorial batches of 20 to 50 URLs.
   - **Deep Queue Champion (LightGBM):** Achieves **0.6872 ROC-AUC** and **81.60% Precision@500**, sustaining high precision across large monthly audit queues and averting an estimated 52 false-positive rewrites relative to the rule baseline.
2. **The Memorization Gap (+0.0832 AUC Inflation):**
   - Standard random K-Fold CV creates an artificial **+0.0832 ROC-AUC** and **+10.00% Precision@50** inflation by memorizing client publishing templates. True out-of-domain generalizability must be evaluated under client-grouped splits (`GroupKFold` on `client_id`).
3. **Audited SEO Dogmas:**
   - **Content Age:** Post-launch content (31–90d) experiences higher decay (**66.87%**) than legacy evergreen content (>365d: **42.63%**, $h=0.495$, `OPPOSITE`).
   - **Word Count:** Articles $>3,500$ words decay at **59.68%** vs. **20.66%** for short navigational content ($h=0.822$, `OPPOSITE`).
   - **Position Tier:** Striking distance (pos 11–20) suffers the highest decay across all tiers (**60.95%** vs. Top 3: **24.08%**, $h=0.763$, `CONFIRMED`).

---

## Project Structure

```text
├── data/
│   ├── raw/
│   │   └── content_refresh_anonymized.csv  <- 30,000 mature URLs across 32 clients
│   └── processed/
│       └── refresh_feature_vector.csv      <- Engineered feature vectors (44 -> 50 cols)
├── docs/
│   ├── index.html                          <- Interactive web report
│   ├── research_paper.md                   <- Publication-grade peer-reviewed paper
│   ├── data-dictionary.md                  <- 44-field specification & platform quirks
│   └── img/                                <- Publication figures (Figures 1, 2, 3)
├── notebooks/
│   ├── capstone_content_decay.ipynb        <- Master Executive Capstone Notebook
│   ├── 01_research_question.ipynb          <- Problem framing & base rate (54.21%)
│   ├── 02_data_contract.ipynb              <- Schema & allowable range assertions
│   ├── 03_signal_audit.ipynb               <- Cohen's h effect sizes & decision register
│   ├── 04_baseline_model.ipynb             <- Simple sort & composite rule baselines
│   ├── 05_model_training.ipynb             <- 5-Fold GroupKFold CV & Dual-Model benchmark
│   ├── 06_leakage_and_validation.ipynb     <- Confession test & Memorization Gap
│   └── 07_honest_claims.ipynb              <- Diversified playbook & dynamic charts
├── outputs/
│   ├── model_comparison_scorecard.csv      <- Benchmark metrics across all cutoffs
│   ├── validation_memorization_gap.csv     <- Random vs. Grouped CV inflation metrics
│   ├── editorial_action_playbook_top100.csv<- Prioritized review queue
│   ├── rf_prioritized_queue_top100.csv     <- Random Forest top queue
│   └── lgb_prioritized_queue_top100.csv    <- LightGBM top queue
└── scripts/
    ├── ml_utils.py                         <- Shared feature constants and metrics
    ├── 01_prepare_features.py              <- Data contract assertions & feature prep
    ├── 02_train_models.py                  <- 5-fold CV training loop & export
    └── run_all.py                          <- Single-command master pipeline runner
```

---

## Reproducibility

```bash
# Clone and enter directory
git clone <repo-url>
cd "internship project"

# Install virtual environment
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt  # or run using the configured environment

# Run master automated pipeline
python scripts/run_all.py
```
