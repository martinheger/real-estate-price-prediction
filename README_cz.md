# Predikce cen bytů v Praze — ML ensemble

**[English](README.md) | [Česky](README_cz.md)**

Predikce prodejní ceny bytů v Praze na základě lokality, dispozice, stavu a blízkosti občanské vybavenosti. Finální model kombinuje tři gradient boosting algoritmy do ensemble a dosahuje 9,47 % MAPE.

## Výsledky

Průměr z 10 nezávislých validačních běhů (±směrodatná odchylka):

| Model | MAPE |
|---|---|
| **Stacking ensemble** | **9,47 % ±0,48** |
| Voting ensemble (optimalizované váhy) | 9,48 % ±0,49 |
| XGBoost (Optuna tuning) | 9,51 % ±0,47 |
| LightGBM (Optuna tuning) | 9,64 % ±0,50 |
| Voting ensemble (rovnoměrné váhy) | 9,64 % ±0,50 |
| CatBoost (Optuna tuning) | 9,72 % ±0,52 |
| Random Forest (Optuna tuning) | 10,75 % ±0,53 |

Detailní tabulka po jednotlivých bězích viz [results/report_hw9.txt](./results/report_hw9.txt).

### Poznámka k ensemblu

Stacking překonal samotný XGBoost o 0,046 p.b. a vyhrál v 7 z 10 běhů. Párový t-test dává p = 0,13, takže rozdíl **není statisticky průkazný** — při daném rozptylu mezi běhy (±0,48) by k prokázání zlepšení tohoto řádu bylo potřeba výrazně víc opakování. Ensemble je zvolen jako finální model, protože je konzistentně mírně lepší a stabilnější, ne proto, že by rozdíl byl významný.

Optimalizace vah oproti rovnoměrnému průměru pomáhá měřitelně (9,48 % vs. 9,64 %). Průměrné váhy napříč běhy: XGBoost 0,40 ±0,05, CatBoost 0,46 ±0,07, LightGBM 0,14 ±0,09.

Random Forest dostal ve všech deseti bězích váhu přesně nula. Jeho predikce jsou silně korelované s gradient boostingem a zároveň méně přesné, takže do ensemble nepřidává nic. V pipeline zůstává pouze pro srovnání.

Lineární model (Ridge) do ensemble přidán nebyl. Jeho chyby jsou s gradient boostingem méně korelované (0,82 vs. 0,98 u LightGBM), ale vlastní MAPE kolem 11 % je příliš vysoká na to, aby kombinace přinesla měřitelné zlepšení.

## Co nejvíc ovlivňuje cenu

Podle feature importance z XGBoost modelu:

1. **Typ konstrukce** (panelová vs. cihlová) — 0,31, zdaleka nejsilnější faktor
2. **Plocha bytu** — 0,14
3. **Počet pokojů** — 0,07
4. **Čistá plocha** (bez balkonu a sklepa) — 0,035
5. **Stav bytu** (nový / po rekonstrukci) a **typ vlastnictví** (družstevní vs. osobní) — 0,034

Menší, ale zajímavý vliv mají textové signály přímo z inzerátu (klíčová slova typu luxus, novostavba, družstvo) a lokalita (vzdálenost od centra, blízkost občanské vybavenosti).

## Metodika

- **Data:** 5000 inzerátů bytů v Praze, 32 proměnných (dispozice, lokalita, GPS, POI vzdálenosti, textový popis)
- **Feature engineering:** parsování textového popisu inzerátu, segmentace podle čtvrti a typu konstrukce, vzdálenostní metriky k centru a Hradu, čistá plocha, relativní patro
- **Cílová proměnná:** `log1p(price)`; MAPE se počítá až po zpětné transformaci na koruny
- **Hyperparameter tuning:** Optuna, TPE sampler, 18 trials na model — počet omezen výpočetním časem, nikoli konvergencí
- **Ensembling:** vážený voting (váhy hledané SLSQP solverem) + stacking regressor s RidgeCV meta-modelem

## Validace

Pipeline používá dvě oddělené validační smyčky.

**Ladění hyperparametrů** — 5-fold cross-validace uvnitř Optuny. Odstranění outlierů i preprocessing (target encoding, škálování) probíhá zvlášť v každém foldu a fituje se pouze na jeho trénovací části; validační fold zůstává nedotčený. Early stopping má vlastní oddělenou sadu.

**Odhad výsledku** — 10 opakovaných náhodných hold-out splitů. Reportovaná MAPE a směrodatná odchylka pocházejí odsud, ne z ladicí cross-validace: skóre, podle kterého se vybíraly hyperparametry, je z principu optimistické, protože jde o nejlepší z 18 pokusů. Pro srovnání: ladicí CV hlásí u XGBoostu 9,73 %, hold-out validace 9,51 %.

Rozdělení dat v jednom validačním běhu:

```text
train_raw (5000)
├── 80 % train_d
│   ├── 5-fold CV uvnitř → out-of-fold predikce pro všech 4000 řádků
│   │   → z nich se počítají váhy ensemble a meta-model stackingu
│   └── měřené modely trénují na celých 80 %
│       (počet stromů převzatý z OOF foldů, bez early stoppingu)
└── 20 % val_d ........ pouze měření, nefituje se na něm nic
```

Důvod pro out-of-fold predikce: počet stromů, váhy ensemble i koeficienty meta-modelu jsou parametry učené z dat. Kdyby se učily na validační sadě, byla by výsledná MAPE nadhodnocená. Zároveň ale nebylo žádoucí kvůli nim ubírat trénovací data — proto se odhadnou z vnitřní cross-validace a modely, na kterých se měří, trénují na celé trénovací části.

Odstranění outlierů (0,3 % nejextrémnějších hodnot ceny za m² z obou stran) probíhá výhradně na trénovacích datech. Validační sada zůstává v původním rozložení včetně netypických nabídek.

Interní validace je oproti finálnímu modelu mírně pesimistická — modely v ní trénují na 4000 řádcích, zatímco finální model na všech 5000. Naměřený efekt velikosti trénovací sady je zhruba 0,3 p.b. na 1000 řádků.

## Struktura repa

```text
├── notebooks/
│   └── appartments_prediction_fin.ipynb   # kompletní pipeline: data → tuning → ensembling → report
└── results/
    ├── submission_OPTUNA_STACKING.csv     # finální predikce na testovacích datech
    └── report_hw9.txt                     # detailní výsledky validace a feature importance
```

Notebook si trénovací i testovací data stahuje sám při spuštění, není tedy potřeba žádný lokální dataset. Celý běh trvá zhruba 2,5 hodiny na CPU.

## Tech stack

Python, XGBoost, LightGBM, CatBoost, Optuna, scikit-learn

## Autorství a kontext

Projekt vznikl v rámci kurzu PřF:M7DataSP – Praktikum z pokročilé datové vědy (2025) na Masarykově univerzitě, na základě course repozitáře [simecek/dspracticum2024](https://github.com/simecek/dspracticum2024) a sdíleného týmového repozitáře [LuciaKajanova/dspracticum25_flowers_team](https://github.com/LuciaKajanova/dspracticum25_flowers_team).

Oficiálně šlo o skupinové zadání (spolupráce s Lucia Kajanová a Eva Kopřivová), kompletní ML pipeline v tomto repozitáři je ale moje individuální práce.

## Licence

MIT — viz [LICENSE](./LICENSE)

