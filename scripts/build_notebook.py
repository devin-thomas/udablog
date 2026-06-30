from __future__ import annotations

import inspect
import sys
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def markdown_cell(text: str) -> nbf.NotebookNode:
    """Create a markdown cell."""
    return nbf.v4.new_markdown_cell(inspect.cleandoc(text))


def code_cell(code: str) -> nbf.NotebookNode:
    """Create a code cell."""
    return nbf.v4.new_code_cell(inspect.cleandoc(code))


def build_notebook() -> nbf.NotebookNode:
    """Assemble the project notebook with CRISP-DM structure and business questions."""
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        markdown_cell(
            """
            # What Really Predicts GDP per Capita?

            ## CRISP-DM framing

            This notebook uses official World Bank indicators to answer four business questions:

            1. Which public indicators matter most when predicting GDP per capita?
            2. How do the profiles of low- and high-GDP countries differ?
            3. How accurate is a machine learning model on unseen countries?
            4. What happens in a creative "digital uplift" scenario?

            The audience for the supporting blog post is non-technical, but this notebook keeps the full technical trail for the project submission.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys

            project_root = Path.cwd().resolve().parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            import json
            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import Image, display

            from src.world_bank_pipeline import (
                configure_plotting,
                get_profile_gap_table,
                get_region_summary,
                load_or_build_dataset,
                plot_cover_chart,
                plot_feature_importance,
                plot_prediction_scatter,
                plot_region_summary,
                plot_scenario_summary,
                train_final_model,
            )

            configure_plotting()
            pd.set_option("display.max_columns", 50)
            """
        ),
        markdown_cell(
            """
            ## Business understanding

            GDP per capita is not the whole story of development, but it is a strong proxy for the economic resources available to residents. I frame this analysis as a public-data triage exercise: if a policymaker only has a short list of indicators, which ones carry the most signal about national prosperity?
            """
        ),
        code_cell(
            """
            dataset = load_or_build_dataset(force_refresh=False)
            dataset.head()
            """
        ),
        markdown_cell(
            """
            ## Data understanding

            The dataset covers 2015-2023 and combines GDP per capita with digital access, electricity access, life expectancy, trade openness, unemployment, inflation, foreign direct investment, education spending, population, urbanization, and region.
            """
        ),
        code_cell(
            """
            pd.DataFrame(
                {
                    "rows": [dataset.shape[0]],
                    "countries": [dataset["country_name"].nunique()],
                    "years": [dataset["year"].nunique()],
                }
            )
            """
        ),
        code_cell(
            """
            missing_summary = (
                dataset.isna()
                .mean()
                .sort_values(ascending=False)
                .rename("missing_share")
                .to_frame()
            )
            missing_summary.head(10)
            """
        ),
        markdown_cell(
            """
            ## Data preparation

            The modeling pipeline handles missing numeric values with median imputation, missing categorical values with the most frequent category, and region with one-hot encoding. I keep the GDP target on its original dollar scale for reporting, but the model trains on a log-transformed target to reduce the influence of extreme outliers.
            """
        ),
        markdown_cell(
            """
            ## Question 1: Which indicators matter most?
            """
        ),
        code_cell(
            """
            model_bundle = train_final_model(dataset)
            model_bundle.feature_importance.head(10)
            """
        ),
        code_cell(
            """
            feature_fig = plot_feature_importance(model_bundle.feature_importance)
            plt.show()
            """
        ),
        markdown_cell(
            """
            The strongest drivers are the country-level development basics: internet access, life expectancy, electricity access, and trade openness. That pattern matters because it points to broad capability-building indicators, not narrow short-term levers.
            """
        ),
        markdown_cell(
            """
            ## Question 2: How do the profiles differ across GDP tiers?
            """
        ),
        code_cell(
            """
            profile_gap = get_profile_gap_table(dataset)
            profile_gap
            """
        ),
        code_cell(
            """
            region_summary = get_region_summary(dataset)
            region_fig = plot_region_summary(region_summary)
            plt.show()
            """
        ),
        markdown_cell(
            """
            The highest-GDP countries are not just richer in the abstract. They also show much higher internet penetration, stronger electricity access, longer life expectancy, and more outward-facing trade systems.
            """
        ),
        markdown_cell(
            """
            ## Question 3: How accurate is the model?
            """
        ),
        code_cell(
            """
            model_bundle.comparison
            """
        ),
        code_cell(
            """
            pd.DataFrame([model_bundle.metrics]).round(2)
            """
        ),
        code_cell(
            """
            prediction_fig = plot_prediction_scatter(model_bundle.predictions)
            plt.show()
            """
        ),
        markdown_cell(
            """
            I evaluate models with grouped splits so an individual country never appears in both training and validation. That is stricter than a random row split and makes the accuracy claim more credible. The best model is selected by cross-validated RMSE and then tested on a held-out set of unseen countries.
            """
        ),
        markdown_cell(
            """
            ## Question 4: What happens in a creative predictive scenario?
            """
        ),
        code_cell(
            """
            pd.DataFrame(
                {
                    "scenario": ["Baseline", "Digital uplift"],
                    "country_or_profile": [
                        model_bundle.scenario_summary["baseline_country"],
                        "Same baseline with stronger infrastructure",
                    ],
                    "gdp_per_capita_usd": [
                        model_bundle.scenario_summary["baseline_actual_gdp_usd"],
                        model_bundle.scenario_summary["baseline_prediction_usd"],
                    ],
                    "predicted_gdp_per_capita_usd": [
                        model_bundle.scenario_summary["baseline_prediction_usd"],
                        model_bundle.scenario_summary["uplift_prediction_usd"],
                    ],
                }
            ).round(0)
            """
        ),
        code_cell(
            """
            scenario_fig = plot_scenario_summary(model_bundle.scenario_summary)
            plt.show()
            """
        ),
        markdown_cell(
            """
            The scenario starts from the real 2023 profile of the Sub-Saharan African country that sits closest to the regional median, then lifts internet access, electricity access, education spending, trade openness, and life expectancy toward the middle-to-upper global GDP band. This is not a causal forecast. It is a directional model-based estimate that shows how much economic signal is tied up in these development fundamentals.
            """
        ),
        markdown_cell(
            """
            ## Conclusion

            The project answer is straightforward: GDP per capita is most predictable where basic systems work. Countries that combine reliable electricity, broad internet access, better health outcomes, and stronger trade integration tend to cluster at much higher income levels. A supervised learning model can capture that pattern with useful accuracy, and the scenario exercise shows why infrastructure and access indicators are practical signals for decision-makers.
            """
        ),
        code_cell(
            """
            cover_fig = plot_cover_chart(dataset)
            plt.show()
            """
        ),
    ]
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.14",
        },
    }
    return notebook


def main() -> None:
    """Write the generated notebook to disk."""
    notebook = build_notebook()
    output_path = PROJECT_ROOT / "notebooks" / "world_bank_gdp_analysis.ipynb"
    output_path.write_text(nbf.writes(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()
