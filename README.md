# Transfer learning for tourism-demand policy: replication code

Replication code for the final chapter of the doctoral thesis

> Abdous, S. (2025). *Determinants of International Tourism Demand for Algeria: An Econometric Study (2000–2020)*. University of Tipaza (then Centre Universitaire Morsli Abdellah), Algeria. https://doi.org/10.5281/zenodo.23572026

The chapter asks whether models of tourism demand fitted on four comparator economies (Egypt, Jordan, Morocco, Tunisia) can be transferred to Algeria, which has very few observations, using two domain-adaptation methods: **KLIEP** importance weighting (instance-based) and **CORAL** feature alignment (feature-based), both from the [ADAPT](https://adapt-python.github.io/adapt/) library, with a Random Forest regressor as the base learner. It then builds counterfactual policy scenarios for Algeria to 2032.

## Results in one paragraph

Transfer did **not** work in the usual sense: every source country gave a negative out-of-sample R² on Algerian data (best: Jordan, KLIEP, R² = −10.8; CORAL was worse everywhere). With twenty annual observations per country and large structural differences between economies, standard adaptation methods are not enough — the thesis reports this as a negative finding and as an open problem, which is the starting point of my current research on reliable cross-country transfer with calibrated uncertainty. Full printed outputs are in `results/`.

## Contents

| File | What it does |
|---|---|
| `src/01_coral_transfer.py` | CORAL alignment + Random Forest, one source country at a time; comparison table and plots |
| `src/02a_kliep_transfer.py` | KLIEP importance weighting + Random Forest, one source country at a time |
| `src/02_coral_kliep_transfer.py` | KLIEP and CORAL side by side; performance table on the Algerian target |
| `src/03_counterfactual_scenarios.py` | Extrapolates Algerian features and produces baseline / policy scenario forecasts to 2032 |
| `results/*_output.txt` | Console output of each script as run for the thesis |
| `data/` | Input data: `ind.xlsx` (5-country tourism panel used by the transfer scripts) and `panel_26_countries_2000_2020.xlsx` (the main gravity panel of the thesis), see `data/README.md` |

## How to run

```bash
pip install -r requirements.txt
python src/01_coral_transfer.py
python src/02a_kliep_transfer.py
python src/02_coral_kliep_transfer.py
python src/03_counterfactual_scenarios.py
```

Scripts expect `data/ind.xlsx` in the repository root layout above. Plots open in a window (`plt.show()`); in Google Colab they render inline.

## Authorship and credit

The research design, the choice of methods, the data collection and cleaning, and the interpretation of all results are mine. The scripts in `src/` were implemented with the technical support of **Dr. Marouane Roudane**, who wrote them with me during the thesis; they are published here with his agreement, unchanged except for file paths and the separation of code from console output. The other empirical chapters of the thesis (panel estimators, time-varying coefficients, classification by competitiveness tier) were run in Stata, R (`tvReg`) and Python; their scripts will be added as they are cleaned up.

## Citation

If you use this code, please cite the thesis (DOI above). Code licence: MIT. Data: see `data/README.md` for sources and terms.

— Samira Abdous · samira.abdous.analyst@gmail.com · ORCID 0009-0006-5310-4884
