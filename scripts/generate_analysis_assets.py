from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.world_bank_pipeline import (  # noqa: E402
    IMAGES_DIR,
    configure_plotting,
    get_profile_gap_table,
    get_region_summary,
    load_or_build_dataset,
    plot_cover_chart,
    plot_feature_importance,
    plot_prediction_scatter,
    plot_region_summary,
    plot_scenario_summary,
    save_figure,
    train_final_model,
)


def main() -> None:
    """Build the dataset snapshot, model outputs, figures, and summary JSON."""
    configure_plotting()
    dataset = load_or_build_dataset(force_refresh=True)
    model_bundle = train_final_model(dataset)
    profile_gap = get_profile_gap_table(dataset)
    region_summary = get_region_summary(dataset)

    save_figure(plot_cover_chart(dataset), IMAGES_DIR / "cover.png")
    save_figure(
        plot_feature_importance(model_bundle.feature_importance),
        IMAGES_DIR / "feature_importance.png",
    )
    save_figure(
        plot_prediction_scatter(model_bundle.predictions),
        IMAGES_DIR / "prediction_scatter.png",
    )
    save_figure(
        plot_scenario_summary(model_bundle.scenario_summary),
        IMAGES_DIR / "scenario_summary.png",
    )
    save_figure(plot_region_summary(region_summary), IMAGES_DIR / "region_summary.png")

    summary = {
        "dataset_rows": int(dataset.shape[0]),
        "dataset_countries": int(dataset["country_name"].nunique()),
        "dataset_years": sorted(dataset["year"].unique().tolist()),
        "model_name": model_bundle.model_name,
        "metrics": model_bundle.metrics,
        "top_features": model_bundle.feature_importance.head(5).to_dict(orient="records"),
        "profile_gap_table": profile_gap.to_dict(orient="records"),
        "scenario_summary": model_bundle.scenario_summary,
        "region_summary": region_summary.to_dict(orient="records"),
    }

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    profile_gap.to_csv(output_dir / "profile_gap_table.csv", index=False)
    region_summary.to_csv(output_dir / "region_summary.csv", index=False)
    pd.DataFrame([model_bundle.metrics]).to_csv(output_dir / "model_metrics.csv", index=False)


if __name__ == "__main__":
    main()
