from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "world_bank_gdp_panel.csv"
IMAGES_DIR = PROJECT_ROOT / "images"

WORLD_BANK_BASE = "https://api.worldbank.org/v2"
START_YEAR = 2015
END_YEAR = 2023

INDICATORS = {
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "IT.NET.USER.ZS": "internet_users_pct",
    "EG.ELC.ACCS.ZS": "electricity_access_pct",
    "SP.DYN.LE00.IN": "life_expectancy_years",
    "SP.URB.TOTL.IN.ZS": "urban_population_pct",
    "NE.TRD.GNFS.ZS": "trade_pct_gdp",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
    "FP.CPI.TOTL.ZG": "inflation_pct",
    "SP.POP.TOTL": "population_total",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_pct_gdp",
    "SE.XPD.TOTL.GD.ZS": "education_spend_pct_gdp",
}

READABLE_COLUMNS = {
    "gdp_per_capita_usd": "GDP per capita (USD)",
    "internet_users_pct": "Internet users (%)",
    "electricity_access_pct": "Electricity access (%)",
    "life_expectancy_years": "Life expectancy (years)",
    "urban_population_pct": "Urban population (%)",
    "trade_pct_gdp": "Trade (% of GDP)",
    "unemployment_pct": "Unemployment (%)",
    "inflation_pct": "Inflation (%)",
    "population_total": "Population",
    "fdi_pct_gdp": "FDI inflows (% of GDP)",
    "education_spend_pct_gdp": "Education spend (% of GDP)",
    "region": "Region",
    "year": "Year",
}


@dataclass
class ModelBundle:
    model_name: str
    pipeline: TransformedTargetRegressor
    metrics: dict[str, float]
    comparison: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    baseline_scenario: pd.Series
    uplift_scenario: pd.Series
    scenario_summary: dict[str, float]


def configure_plotting() -> None:
    """Apply a consistent chart style for notebook, README, and blog assets."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (12, 7)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.titlepad"] = 18
    plt.rcParams["axes.labelpad"] = 10


def _fetch_world_bank_json(endpoint: str, params: dict[str, object] | None = None) -> list[dict]:
    """Fetch every page from a World Bank API endpoint and return the records."""
    session = requests.Session()
    page = 1
    records: list[dict] = []
    while True:
        request_params = {
            "format": "json",
            "per_page": 20000,
            "page": page,
        }
        if params:
            request_params.update(params)
        response = session.get(f"{WORLD_BANK_BASE}{endpoint}", params=request_params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if len(payload) < 2:
            break
        metadata, page_records = payload
        records.extend(page_records)
        total_pages = int(metadata["pages"])
        if page >= total_pages:
            break
        page += 1
    return records


def fetch_country_metadata() -> pd.DataFrame:
    """Download country metadata and keep only fields used in the analysis."""
    records = _fetch_world_bank_json("/country")
    metadata = pd.DataFrame(
        {
            "country_code": [record["id"] for record in records],
            "country_name": [record["name"] for record in records],
            "region": [record["region"]["value"] for record in records],
            "income_level": [record["incomeLevel"]["value"] for record in records],
            "lending_type": [record["lendingType"]["value"] for record in records],
        }
    )
    metadata["region"] = metadata["region"].str.strip()
    metadata["income_level"] = metadata["income_level"].str.strip()
    metadata["lending_type"] = metadata["lending_type"].str.strip()
    return metadata.loc[metadata["region"] != "Aggregates"].reset_index(drop=True)


def fetch_indicator_panel(indicator_code: str, column_name: str) -> pd.DataFrame:
    """Download one indicator series and return a tidy dataframe."""
    records = _fetch_world_bank_json(
        f"/country/all/indicator/{indicator_code}",
        params={"date": f"{START_YEAR}:{END_YEAR}"},
    )
    frame = pd.DataFrame(
        {
            "country_code": [record["countryiso3code"] for record in records],
            "country_name": [record["country"]["value"] for record in records],
            "year": [int(record["date"]) for record in records],
            column_name: [record["value"] for record in records],
        }
    )
    return frame.loc[frame["country_code"] != ""].reset_index(drop=True)


def build_world_bank_dataset() -> pd.DataFrame:
    """Build the full modeling dataset from country metadata and selected indicators."""
    metadata = fetch_country_metadata()
    merged: pd.DataFrame | None = None
    for indicator_code, column_name in INDICATORS.items():
        indicator_frame = fetch_indicator_panel(indicator_code, column_name)
        if merged is None:
            merged = indicator_frame
        else:
            merged = merged.merge(
                indicator_frame,
                on=["country_code", "country_name", "year"],
                how="outer",
            )
    assert merged is not None
    dataset = merged.merge(metadata, on=["country_code", "country_name"], how="left")
    dataset = dataset.loc[dataset["region"].notna()].copy()
    dataset["year"] = dataset["year"].astype(int)
    dataset = dataset.sort_values(["country_name", "year"]).reset_index(drop=True)
    return dataset


def load_or_build_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """Load the committed dataset snapshot or rebuild it from the API."""
    if DATA_PATH.exists() and not force_refresh:
        return pd.read_csv(DATA_PATH)
    dataset = build_world_bank_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATA_PATH, index=False)
    return dataset


def get_feature_columns(dataset: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature columns used in the model."""
    numeric_features = [
        column
        for column in INDICATORS.values()
        if column != "gdp_per_capita_usd"
    ] + ["year"]
    categorical_features = ["region"]
    missing_numeric = [column for column in numeric_features if column not in dataset.columns]
    if missing_numeric:
        raise ValueError(f"Missing expected numeric columns: {missing_numeric}")
    return numeric_features, categorical_features


def get_modeling_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Filter the panel down to rows that are valid for supervised learning."""
    modeling_dataset = dataset.loc[
        dataset["gdp_per_capita_usd"].notna() & (dataset["gdp_per_capita_usd"] > 0)
    ].copy()
    return modeling_dataset.reset_index(drop=True)


def build_preprocessor(
    numeric_features: Iterable[str], categorical_features: Iterable[str]
) -> ColumnTransformer:
    """Create the shared preprocessing pipeline for every model candidate."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_features)),
            ("cat", categorical_pipeline, list(categorical_features)),
        ]
    )


def build_candidate_models(
    numeric_features: Iterable[str], categorical_features: Iterable[str]
) -> dict[str, TransformedTargetRegressor]:
    """Define the candidate regressors evaluated during model selection."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    candidates = {
        "Linear regression": LinearRegression(),
        "Gradient boosting": GradientBoostingRegressor(random_state=42),
        "Random forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }
    wrapped_models: dict[str, TransformedTargetRegressor] = {}
    for name, regressor in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor),
            ]
        )
        wrapped_models[name] = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    return wrapped_models


def evaluate_candidates(dataset: pd.DataFrame) -> pd.DataFrame:
    """Run grouped cross-validation so the same country is never in train and validation."""
    modeling_dataset = get_modeling_dataset(dataset)
    numeric_features, categorical_features = get_feature_columns(modeling_dataset)
    models = build_candidate_models(numeric_features, categorical_features)
    X = modeling_dataset[numeric_features + categorical_features]
    y = modeling_dataset["gdp_per_capita_usd"]
    groups = modeling_dataset["country_name"]
    splitter = GroupKFold(n_splits=5)
    rows = []
    for model_name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            groups=groups,
            cv=splitter,
            scoring={
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
            },
            n_jobs=1,
        )
        rows.append(
            {
                "model": model_name,
                "cv_r2_mean": float(np.mean(scores["test_r2"])),
                "cv_mae_mean": float(-np.mean(scores["test_mae"])),
                "cv_rmse_mean": float(-np.mean(scores["test_rmse"])),
            }
        )
    return pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)


def train_final_model(dataset: pd.DataFrame) -> ModelBundle:
    """Select the best model, fit it, and compute holdout metrics plus scenario analysis."""
    modeling_dataset = get_modeling_dataset(dataset)
    comparison = evaluate_candidates(modeling_dataset)
    best_model_name = comparison.iloc[0]["model"]
    numeric_features, categorical_features = get_feature_columns(modeling_dataset)
    models = build_candidate_models(numeric_features, categorical_features)
    model = clone(models[best_model_name])
    X = modeling_dataset[numeric_features + categorical_features]
    y = modeling_dataset["gdp_per_capita_usd"]
    groups = modeling_dataset["country_name"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(splitter.split(X, y, groups=groups))
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "holdout_r2": float(r2_score(y_test, predictions)),
        "holdout_mae": float(mean_absolute_error(y_test, predictions)),
        "holdout_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "holdout_mape": float(
            np.mean(np.abs((y_test - predictions) / y_test.replace(0, np.nan))) * 100
        ),
    }
    predictions_frame = X_test.copy()
    predictions_frame["actual_gdp_per_capita_usd"] = y_test.to_numpy()
    predictions_frame["predicted_gdp_per_capita_usd"] = predictions
    predictions_frame["prediction_error_usd"] = (
        predictions_frame["predicted_gdp_per_capita_usd"]
        - predictions_frame["actual_gdp_per_capita_usd"]
    )
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=15,
        random_state=42,
        n_jobs=1,
        scoring="neg_root_mean_squared_error",
    )
    feature_importance = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": importance.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    baseline_scenario, uplift_scenario, scenario_summary = build_scenarios(dataset, model)
    return ModelBundle(
        model_name=best_model_name,
        pipeline=model,
        metrics=metrics,
        comparison=comparison,
        predictions=predictions_frame,
        feature_importance=feature_importance,
        baseline_scenario=baseline_scenario,
        uplift_scenario=uplift_scenario,
        scenario_summary=scenario_summary,
    )


def build_scenarios(
    dataset: pd.DataFrame, fitted_model: TransformedTargetRegressor
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """Create a baseline and an uplift scenario for the notebook and blog post."""
    modeling_dataset = get_modeling_dataset(dataset)
    latest = modeling_dataset.loc[
        modeling_dataset["year"] == modeling_dataset["year"].max()
    ].copy()
    scenario_features = [
        "internet_users_pct",
        "electricity_access_pct",
        "life_expectancy_years",
        "urban_population_pct",
        "trade_pct_gdp",
        "unemployment_pct",
        "inflation_pct",
        "population_total",
        "fdi_pct_gdp",
        "education_spend_pct_gdp",
    ]
    regional_pool = latest.loc[latest["region"] == "Sub-Saharan Africa"].dropna(
        subset=scenario_features + ["gdp_per_capita_usd"]
    )
    regional_median = regional_pool[scenario_features].median()
    regional_scale = regional_pool[scenario_features].std().replace(0, 1)
    distances = ((regional_pool[scenario_features] - regional_median) / regional_scale).pow(2).sum(axis=1)
    baseline_row = regional_pool.loc[distances.idxmin()]

    lower_cut = latest["gdp_per_capita_usd"].quantile(0.50)
    upper_cut = latest["gdp_per_capita_usd"].quantile(0.75)
    benchmark_pool = latest.loc[
        latest["gdp_per_capita_usd"].between(lower_cut, upper_cut)
    ].dropna(subset=scenario_features)
    benchmark = benchmark_pool[scenario_features].median()

    baseline = pd.Series(
        {
            **baseline_row[scenario_features].to_dict(),
            "year": float(modeling_dataset["year"].max()),
            "region": baseline_row["region"],
        }
    )
    uplift = baseline.copy()
    uplift["internet_users_pct"] = benchmark["internet_users_pct"]
    uplift["electricity_access_pct"] = benchmark["electricity_access_pct"]
    uplift["education_spend_pct_gdp"] = benchmark["education_spend_pct_gdp"]
    uplift["trade_pct_gdp"] = benchmark["trade_pct_gdp"]
    uplift["life_expectancy_years"] = benchmark["life_expectancy_years"]

    scenario_frame = pd.DataFrame([baseline, uplift])
    scenario_predictions = fitted_model.predict(scenario_frame)
    summary = {
        "baseline_country": str(baseline_row["country_name"]),
        "baseline_actual_gdp_usd": float(baseline_row["gdp_per_capita_usd"]),
        "baseline_prediction_usd": float(scenario_predictions[0]),
        "uplift_prediction_usd": float(scenario_predictions[1]),
        "uplift_delta_usd": float(scenario_predictions[1] - scenario_predictions[0]),
        "uplift_delta_pct": float(
            (scenario_predictions[1] - scenario_predictions[0]) / scenario_predictions[0] * 100
        ),
    }
    return baseline, uplift, summary


def get_profile_gap_table(dataset: pd.DataFrame) -> pd.DataFrame:
    """Compare the top and bottom GDP quartiles using the main driver indicators."""
    modeling_dataset = get_modeling_dataset(dataset)
    latest = modeling_dataset.loc[
        modeling_dataset["year"] == modeling_dataset["year"].max()
    ].copy()
    latest["gdp_group"] = pd.qcut(
        latest["gdp_per_capita_usd"],
        q=[0, 0.25, 0.75, 1],
        labels=["Bottom quartile", "Middle 50%", "Top quartile"],
    )
    focus_columns = [
        "internet_users_pct",
        "electricity_access_pct",
        "life_expectancy_years",
        "trade_pct_gdp",
        "education_spend_pct_gdp",
    ]
    grouped = (
        latest.groupby("gdp_group", observed=True)[focus_columns]
        .median()
        .rename(columns=READABLE_COLUMNS)
        .T.reset_index()
        .rename(columns={"index": "indicator"})
    )
    return grouped


def get_region_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    """Summarize GDP per capita by region using the latest year."""
    modeling_dataset = get_modeling_dataset(dataset)
    latest = modeling_dataset.loc[
        modeling_dataset["year"] == modeling_dataset["year"].max()
    ].copy()
    summary = (
        latest.groupby("region", observed=True)["gdp_per_capita_usd"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )
    summary["gdp_per_capita_usd"] = summary["gdp_per_capita_usd"].round(0)
    return summary


def format_usd(value: float, _: int | None = None) -> str:
    """Format numeric axis values as whole-dollar labels."""
    return f"${value:,.0f}"


def plot_cover_chart(dataset: pd.DataFrame) -> plt.Figure:
    """Build the lead image used in the README and blog post."""
    modeling_dataset = get_modeling_dataset(dataset)
    latest = modeling_dataset.loc[
        modeling_dataset["year"] == modeling_dataset["year"].max()
    ].copy()
    figure, axis = plt.subplots(figsize=(14, 8))
    scatter = axis.scatter(
        latest["internet_users_pct"],
        latest["gdp_per_capita_usd"],
        c=latest["electricity_access_pct"],
        cmap="viridis",
        alpha=0.75,
        s=70,
        edgecolor="white",
        linewidth=0.4,
    )
    axis.set_title("Countries with stronger digital access tend to have much higher GDP per capita")
    axis.set_xlabel("Internet users (% of population)")
    axis.set_ylabel("GDP per capita (USD)")
    axis.yaxis.set_major_formatter(FuncFormatter(format_usd))
    colorbar = figure.colorbar(scatter)
    colorbar.set_label("Electricity access (%)")
    return figure


def plot_feature_importance(feature_importance: pd.DataFrame) -> plt.Figure:
    """Plot the top permutation-importance features."""
    top_features = feature_importance.head(8).copy()
    top_features["label"] = top_features["feature"].map(READABLE_COLUMNS).fillna(top_features["feature"])
    figure, axis = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=top_features,
        x="importance",
        y="label",
        palette="crest",
        hue="label",
        legend=False,
        ax=axis,
    )
    axis.set_title("Internet access, life expectancy, and electricity access carry the most predictive signal")
    axis.set_xlabel("Permutation importance (RMSE increase when shuffled)")
    axis.set_ylabel("")
    return figure


def plot_prediction_scatter(predictions: pd.DataFrame) -> plt.Figure:
    """Compare actual and predicted GDP per capita on the holdout set."""
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=predictions,
        x="actual_gdp_per_capita_usd",
        y="predicted_gdp_per_capita_usd",
        hue="region",
        alpha=0.7,
        ax=axis,
    )
    min_value = min(
        predictions["actual_gdp_per_capita_usd"].min(),
        predictions["predicted_gdp_per_capita_usd"].min(),
    )
    max_value = max(
        predictions["actual_gdp_per_capita_usd"].max(),
        predictions["predicted_gdp_per_capita_usd"].max(),
    )
    axis.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--")
    axis.set_title("Holdout predictions track real GDP per capita well across unseen countries")
    axis.set_xlabel("Actual GDP per capita (USD)")
    axis.set_ylabel("Predicted GDP per capita (USD)")
    axis.xaxis.set_major_formatter(FuncFormatter(format_usd))
    axis.yaxis.set_major_formatter(FuncFormatter(format_usd))
    return figure


def plot_scenario_summary(scenario_summary: dict[str, float]) -> plt.Figure:
    """Compare the baseline and uplift scenario predictions."""
    scenario_frame = pd.DataFrame(
        {
            "scenario": ["Baseline", "Digital uplift"],
            "predicted_gdp_per_capita_usd": [
                scenario_summary["baseline_prediction_usd"],
                scenario_summary["uplift_prediction_usd"],
            ],
        }
    )
    figure, axis = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=scenario_frame,
        x="scenario",
        y="predicted_gdp_per_capita_usd",
        palette=["#4C78A8", "#54A24B"],
        hue="scenario",
        legend=False,
        ax=axis,
    )
    axis.set_title("The uplift scenario nearly doubles predicted GDP per capita")
    axis.set_xlabel("")
    axis.set_ylabel("Predicted GDP per capita (USD)")
    axis.yaxis.set_major_formatter(FuncFormatter(format_usd))
    return figure


def plot_region_summary(region_summary: pd.DataFrame) -> plt.Figure:
    """Show median GDP per capita by region in the latest year."""
    figure, axis = plt.subplots(figsize=(13, 7))
    sns.barplot(
        data=region_summary,
        x="gdp_per_capita_usd",
        y="region",
        palette="mako",
        hue="region",
        legend=False,
        ax=axis,
    )
    axis.set_title("Median GDP per capita remains sharply uneven across regions")
    axis.set_xlabel("Median GDP per capita (USD)")
    axis.set_ylabel("")
    axis.xaxis.set_major_formatter(FuncFormatter(format_usd))
    return figure


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Persist a figure and close it to keep batch scripts memory-safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
