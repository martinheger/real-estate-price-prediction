# Prague Real Estate Price Prediction — ML Ensemble

**[English](README.md) | [Česky](README_cz.md)**

Prediction of apartment sales prices in Prague based on location, layout, condition, and proximity to civic amenities. The final model combines three gradient boosting algorithms into an ensemble, achieving 9.47% MAPE.

## Results

Average over 10 independent validation runs (±standard deviation):

| Model | MAPE |
|---|---|
| **Stacking ensemble** | **9.47% ±0.48** |
| Voting ensemble (optimized weights) | 9.48% ±0.49 |
| XGBoost (Optuna tuning) | 9.51% ±0.47 |
| LightGBM (Optuna tuning) | 9.64% ±0.50 |
| Voting ensemble (equal weights) | 9.64% ±0.50 |
| CatBoost (Optuna tuning) | 9.72% ±0.52 |
| Random Forest (Optuna tuning) | 10.75% ±0.53 |

For a detailed run-by-run table, see [results/report_hw9.txt](./results/report_hw9.txt).

### A note on the ensemble

Stacking outperformed XGBoost alone by 0.046 pp and won in 7 of 10 runs. A paired t-test gives p = 0.13, so the difference is **not statistically significant** — given the between-run variance (±0.48), demonstrating an improvement of this magnitude would require considerably more repetitions. The ensemble is chosen as the final model because it is consistently slightly better and more stable, not because the difference is significant.

Weight optimization does help measurably over a simple average (9.48% vs. 9.64%). Mean weights across runs: XGBoost 0.40 ±0.05, CatBoost 0.46 ±0.07, LightGBM 0.14 ±0.09.

Random Forest received a weight of exactly zero in all ten runs. Its predictions are strongly correlated with the gradient boosting models while being less accurate, so it contributes nothing to the ensemble. It remains in the pipeline for comparison only.

A linear model (Ridge) was not added to the ensemble. Its errors are less correlated with gradient boosting (0.82 vs. 0.98 for LightGBM), but its own MAPE of around 11% is too high for the combination to yield a measurable improvement.

## Main Price Drivers

According to feature importance from the XGBoost model:

1. **Construction type** (panel vs. brick) — 0.31, by far the strongest factor
2. **Apartment area** — 0.14
3. **Number of rooms** — 0.07
4. **Net area** (excluding balcony and cellar) — 0.035
5. **Condition** (new / renovated) and **ownership type** (cooperative vs. private) — 0.034

A smaller but interesting impact comes from text signals in the listing itself (keywords such as luxury, new development, cooperative) and from location (distance to the centre, proximity to amenities).

## Methodology

- **Data:** 5000 apartment listings in Prague, 32 variables (layout, location, GPS, POI distances, text descriptions)
- **Feature engineering:** parsing listing text, segmentation by district and construction type, distance metrics to the city centre and Prague Castle, net area, relative floor
- **Target:** `log1p(price)`; MAPE is computed after transforming predictions back to CZK
- **Hyperparameter tuning:** Optuna, TPE sampler, 18 trials per model — limited by compute budget, not by convergence
- **Ensembling:** weighted voting (weights found via SLSQP solver) + stacking regressor with a RidgeCV meta-model

## Validation

The pipeline uses two separate validation loops.

**Hyperparameter tuning** — 5-fold cross-validation inside Optuna. Outlier removal and preprocessing (target encoding, scaling) are performed separately within each fold and fitted only on that fold's training portion; the validation fold is left untouched. Early stopping uses its own separate set.

**Performance estimate** — 10 repeated random hold-out splits. The reported MAPE and standard deviation come from here, not from the tuning cross-validation: the score used to select hyperparameters is optimistic by construction, being the best of 18 attempts. For comparison, tuning CV reports 9.73% for XGBoost while hold-out validation reports 9.51%.

Data split within a single validation run:

```text
train_raw (5000)
├── 80 % train_d
│   ├── inner 5-fold CV → out-of-fold predictions for all 4000 rows
│   │   → used to fit ensemble weights and the stacking meta-model
│   └── evaluated models are then trained on the full 80 %
│       (tree count carried over from the OOF folds, no early stopping)
└── 20 % val_d ........ measurement only, nothing is fitted on it
```

The reason for out-of-fold predictions: the number of trees, the ensemble weights and the meta-model coefficients are all parameters learned from data. Fitting them on the validation set would inflate the reported MAPE. At the same time, it was undesirable to sacrifice training data for them — hence they are estimated from inner cross-validation while the evaluated models train on the entire training portion.

Outlier removal (the 0.3% most extreme price-per-m² values on each side) is applied to training data only. The validation set retains its original distribution, including atypical listings.

The internal validation is slightly pessimistic relative to the final model — models train on 4000 rows here, whereas the final model uses all 5000. The measured effect of training set size is roughly 0.3 pp per 1000 rows.

## Repo Structure

```text
├── notebooks/
│   └── appartments_prediction_fin.ipynb   # full pipeline: data → tuning → ensembling → report
└── results/
    ├── submission_OPTUNA_STACKING.csv     # final predictions on test data
    └── report_hw9.txt                     # detailed validation results and feature importance
```

The notebook downloads the training and test data automatically upon execution; no local dataset is required. A full run takes roughly 2.5 hours on CPU.

## Tech Stack

Python, XGBoost, LightGBM, CatBoost, Optuna, scikit-learn

## Authorship and Context

Project created for the course PřF:M7DataSP – Advanced Data Science Practicum (2025) at Masaryk University, based on the course repository [simecek/dspracticum2024](https://github.com/simecek/dspracticum2024) and the shared team repository [LuciaKajanova/dspracticum25_flowers_team](https://github.com/LuciaKajanova/dspracticum25_flowers_team).

Officially a group assignment (in collaboration with Lucia Kajanová and Eva Kopřivová), but the complete ML pipeline in this repository is my individual work.

## License

MIT — see [LICENSE](./LICENSE)

