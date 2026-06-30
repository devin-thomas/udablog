# What Really Predicts GDP per Capita?

This project is the `udablog` submission for the Udacity Data Scientist Nanodegree. I use official World Bank indicators from 2015-2023 to study which country-level signals are most associated with GDP per capita and to test whether a supervised learning model can predict GDP per capita for unseen countries.

## Motivation

GDP per capita is not a complete measure of development, but it is a practical proxy for the economic resources available to residents. I wanted to answer four concrete questions:

1. Which public indicators matter most when predicting GDP per capita?
2. How do low- and high-GDP countries differ in their development profiles?
3. How accurate is a machine learning model on countries it has not seen before?
4. What happens in a creative "digital uplift" scenario?

## Key findings

- The best model was a gradient boosting regressor with grouped country validation.
- Holdout performance on unseen countries reached an `R^2` of `0.71`.
- The strongest predictive signals were life expectancy, electricity access, internet usage, unemployment, and urbanization.
- In the latest-year comparison, top-quartile GDP countries had much higher internet usage, stronger electricity access, and longer life expectancy than bottom-quartile countries.
- In the scenario analysis, a Sub-Saharan African baseline profile close to Benin's 2023 position rose from about `$1.5k` predicted GDP per capita to about `$10.9k` under a stronger infrastructure and access profile. This is a directional model estimate, not a causal forecast.

## Repository structure

- `notebooks/world_bank_gdp_analysis.ipynb` - executed notebook with CRISP-DM framing, analysis, modeling, visuals, and conclusions.
- `src/world_bank_pipeline.py` - reusable data collection, preprocessing, modeling, scenario, and plotting functions.
- `scripts/generate_analysis_assets.py` - refreshes the World Bank snapshot, metrics, and figures.
- `scripts/build_notebook.py` - generates the notebook structure programmatically.
- `data/processed/world_bank_gdp_panel.csv` - committed dataset snapshot used by the notebook.
- `data/processed/analysis_summary.json` - compact summary of the model outputs used for publishing.
- `images/` - charts used in the README, Pages site, and blog post.
- `blog/substack_article.md` - publication-ready article draft for Substack.
- `docs/` - lightweight static project page intended for GitHub Pages.

## Libraries used

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `requests`
- `jupyter`
- `nbformat`

## How to run

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Refresh the dataset snapshot and figures:

```powershell
.\.venv\Scripts\python.exe scripts\generate_analysis_assets.py
```

4. Rebuild the notebook if needed:

```powershell
.\.venv\Scripts\python.exe scripts\build_notebook.py
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks\world_bank_gdp_analysis.ipynb
```

## Acknowledgements

- World Bank Open Data and the World Bank API for the country-level indicators.
- Udacity's Data Scientist Nanodegree project prompt and rubric for the project framing.
