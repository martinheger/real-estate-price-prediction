# 🏠 Predikce cen bytů v Praze — ML ensemble (XGBoost + LightGBM + CatBoost)

Predikce prodejní ceny bytů v Praze na základě lokality, dispozice, stavu a blízkosti občanské vybavenosti. Finální model kombinuje tři gradient boosting algoritmy do ensemble a dosahuje **9.36 % MAPE**.

> **Poznámka k autorství:** Projekt byl oficiálně zadán jako týmová práce v rámci kurzu, ale kompletní pipeline v tomto repozitáři (feature engineering, tuning, ensembling, validace) je moje individuální práce.

## Výsledky

| Model | MAPE |
|---|---|
| XGBoost (Optuna tuning) | 9.05 % |
| LightGBM (Optuna tuning) | 9.12 % |
| CatBoost (Optuna tuning) | 9.19 % |
| **Voting ensemble (optimalizované váhy)** | **9.36 %** |
| **Stacking ensemble** | **9.36 %** |

*(Průměr z 10 nezávislých validačních běhů, směrodatná odchylka ±0.44 %. Detailní tabulka viz [results/report_hw9.txt](./results/report_hw9.txt))*

## Co nejvíc ovlivňuje cenu

Podle feature importance z XGBoost modelu:

1. **Typ konstrukce** (panelová vs. cihlová) — zdaleka nejsilnější faktor
2. **Plocha bytu**
3. **Počet pokojů**
4. **Typ vlastnictví** (družstevní vs. osobní)
5. **Stav bytu** (nový / po rekonstrukci)

Menší, ale zajímavý vliv mají textové signály přímo z inzerátu (`luxury`, `new`, `parking` klíčová slova) a lokalita (vzdálenost od centra, MHD, občanská vybavenost).

## Metodika

- **Data:** 5000 inzerátů bytů v Praze, ~35 proměnných (dispozice, lokalita, GPS, POI vzdálenosti, textový popis)
- **Feature engineering:** parsování textového popisu inzerátu, segmentace podle čtvrti a typu konstrukce, vzdálenostní metriky k centru
- **Hyperparameter tuning:** Optuna (18 trials na model)
- **Ensembling:** vážený voting (optimalizované váhy) + stacking regressor
- **Validace:** 10× opakovaný train/validation split pro odhad stability výsledků

## Struktura repa

```
├── notebooks/
│   └── appartments_prediction_fin.ipynb   # kompletní pipeline: data → tuning → ensembling → report
└── results/
    ├── submission_OPTUNA_STACKING.csv     # finální predikce na testovacích datech
    └── report_hw9.txt                     # detailní výsledky validace a feature importance
```

> Notebook si trénovací i testovací data stahuje sám při spuštění (`wget` z veřejného zdroje), není tedy potřeba žádný lokální dataset.

## Tech stack
Python · XGBoost · LightGBM · CatBoost · Optuna · scikit-learn

---
*Školní projekt (Aplikovaná matematika, kurz Machine Learning). Cílová metrika: Mean Absolute Percentage Error (MAPE).*
