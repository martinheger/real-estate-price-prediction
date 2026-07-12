# Prague Real Estate Price Prediction — ML Ensemble

**[English](README.md) | [Česky](README_cz.md)**

Prediction of apartment sales prices in Prague based on location, layout, condition, and proximity to civic amenities. The final model combines three gradient boosting algorithms into an ensemble, achieving a 9.36% MAPE.

## Results

| Model | MAPE |
|---|---|
| XGBoost (Optuna tuning) | 9.05 % |
| LightGBM (Optuna tuning) | 9.12 % |
| CatBoost (Optuna tuning) | 9.19 % |
| Voting ensemble (optimized weights) | 9.36 % |
| Stacking ensemble | 9.36 % |

Average from 10 independent validation runs, standard deviation ±0.44%. For a detailed table, see [results/report_hw9.txt](./results/report_hw9.txt).

## Main Price Drivers

According to feature importance from the XGBoost model:

1. Construction type (panel vs. brick) — by far the strongest factor
2. Apartment area
3. Number of rooms
4. Ownership type (cooperative vs. private)
5. Condition (new / renovated)

A smaller but interesting impact comes from text signals directly in the listing ("luxury", "new", "parking" keywords) and location (distance to center, public transit, civic amenities).

## Methodology

- Data: 5000 apartment listings in Prague, ~35 variables (layout, location, GPS, POI distances, text descriptions)
- Feature engineering: parsing text descriptions, segmentation by district and construction type, distance metrics to the city center
- Hyperparameter tuning: Optuna (18 trials per model)
- Ensembling: weighted voting (optimized weights) + stacking regressor
- Validation: 10x repeated train/validation split to estimate result stability

## Repo Structure

```text
├── notebooks/
│   └── appartments_prediction_fin.ipynb   # full pipeline: data → tuning → ensembling → report
└── results/
    ├── submission_OPTUNA_STACKING.csv     # final predictions on test data
    └── report_hw9.txt                     # detailed validation results and feature importance
```

The notebook downloads the training and test data automatically upon execution; no local dataset is required.

## Tech Stack

Python, XGBoost, LightGBM, CatBoost, Optuna, scikit-learn

## Authorship and Context

Project created for the course PřF:M7DataSP – Advanced Data Science Practicum (2025) at Masaryk University, based on the course repository [simecek/dspracticum2024](https://github.com/simecek/dspracticum2024) and the shared team repository [LuciaKajanova/dspracticum25_flowers_team](https://github.com/LuciaKajanova/dspracticum25_flowers_team).

Officially a group assignment (in collaboration with Lucia Kajanová and Eva Kopřivová), but the complete ML pipeline in this repository is my individual work.

## License

MIT — see [LICENSE](./LICENSE)
