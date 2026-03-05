import logging
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.files import File

from analytics_engine.models import AnalysisRun
from analytics_engine.services.agents import run_dataset_and_visualization_agents
from analytics_engine.services.visualization_service import create_visualizations
from analytics_engine.utils.data_io import load_dataset_frame
from analytics_engine.utils.validators import validate_feature_plan, validate_visualization_plan

logger = logging.getLogger("analytics_engine")


def _quality_scorecard(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    total_cells = max(rows * cols, 1)
    missing_cells = int(df.isna().sum().sum())
    completeness_pct = round((1 - (missing_cells / total_cells)) * 100, 2)

    duplicate_rows = int(df.duplicated().sum())
    duplicate_row_pct = round((duplicate_rows / max(rows, 1)) * 100, 2)

    numeric_df = df.select_dtypes(include=[np.number])
    outlier_count = 0
    total_numeric_cells = 0
    if not numeric_df.empty:
        for column in numeric_df.columns:
            series = numeric_df[column].dropna()
            total_numeric_cells += len(series)
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count += int(((series < lower) | (series > upper)).sum())

    numeric_outlier_pct = round((outlier_count / max(total_numeric_cells, 1)) * 100, 2)

    high_cardinality_columns = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            ratio = df[col].nunique(dropna=True) / max(rows, 1)
            if ratio > 0.5:
                high_cardinality_columns.append(col)

    # Weighted quality score (higher is better).
    overall_score = round(
        (0.45 * completeness_pct)
        + (0.25 * (100 - duplicate_row_pct))
        + (0.20 * (100 - min(numeric_outlier_pct, 100)))
        + (0.10 * (100 - min(len(high_cardinality_columns) * 10, 100))),
        2,
    )

    return {
        "overall_score": overall_score,
        "completeness_pct": completeness_pct,
        "duplicate_row_pct": duplicate_row_pct,
        "numeric_outlier_pct": numeric_outlier_pct,
        "high_cardinality_columns": high_cardinality_columns,
    }


def _dataset_profile(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(),
        "summary_stats": numeric_df.describe().to_dict() if not numeric_df.empty else {},
        "mean": numeric_df.mean(numeric_only=True).to_dict() if not numeric_df.empty else {},
        "median": numeric_df.median(numeric_only=True).to_dict() if not numeric_df.empty else {},
        "std": numeric_df.std(numeric_only=True).fillna(0).to_dict() if not numeric_df.empty else {},
        "sample_rows": df.head(5).replace({np.nan: None}).to_dict(orient="records"),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "quality_scorecard": _quality_scorecard(df),
    }


def _apply_transformations(df: pd.DataFrame, plan: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    transformed = df.copy()
    applied: dict[str, list[str]] = {}

    for column, actions in plan.items():
        if column not in transformed.columns:
            continue

        for action in actions:
            if action == "handle_missing":
                if pd.api.types.is_numeric_dtype(transformed[column]):
                    transformed[column] = transformed[column].fillna(transformed[column].median())
                else:
                    transformed[column] = transformed[column].fillna("Unknown")
            elif action == "normalize" and pd.api.types.is_numeric_dtype(transformed[column]):
                col_min, col_max = transformed[column].min(), transformed[column].max()
                if pd.notna(col_min) and pd.notna(col_max) and col_max != col_min:
                    transformed[column] = (transformed[column] - col_min) / (col_max - col_min)
            elif action == "standardize" and pd.api.types.is_numeric_dtype(transformed[column]):
                mean = transformed[column].mean()
                std = transformed[column].std()
                if pd.notna(std) and std != 0:
                    transformed[column] = (transformed[column] - mean) / std
            elif action == "one_hot_encode":
                encoded = pd.get_dummies(transformed[column], prefix=column, dummy_na=True)
                transformed = pd.concat([transformed.drop(columns=[column]), encoded], axis=1)
            elif action == "drop":
                transformed = transformed.drop(columns=[column])

            applied.setdefault(column, []).append(action)

    return transformed, applied


def analyze_dataset_and_create_run(dataset, user) -> AnalysisRun:
    run = AnalysisRun.objects.create(owner=user, dataset=dataset, status="running")
    try:
        df = load_dataset_frame(dataset.file.path, dataset.file_type)
        profile = _dataset_profile(df)

        dataset.row_count = df.shape[0]
        dataset.column_count = df.shape[1]
        dataset.schema_json = {
            "columns": list(df.columns),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        }
        dataset.summary_json = profile
        dataset.save(update_fields=["row_count", "column_count", "schema_json", "summary_json"])

        raw_feature_plan, raw_viz_plan = run_dataset_and_visualization_agents(profile)
        clean_feature_plan = validate_feature_plan(raw_feature_plan, set(df.columns))

        transformed_df, applied_plan = _apply_transformations(df, clean_feature_plan)
        processed_columns = list(transformed_df.columns)

        output_dir = Path(settings.MEDIA_ROOT) / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"analysis_{run.id}_dataset_{dataset.id}.csv"
        transformed_df.to_csv(output_file, index=False)

        with output_file.open("rb") as f:
            run.transformed_file.save(output_file.name, File(f), save=False)

        clean_viz_plan = validate_visualization_plan(raw_viz_plan, set(processed_columns))
        create_visualizations(
            dataset=dataset,
            analysis_run=run,
            owner=user,
            frame=transformed_df,
            viz_plan=clean_viz_plan,
        )

        run.status = "completed"
        run.dataset_profile = profile
        run.llm_feature_plan = clean_feature_plan
        run.applied_transformations = applied_plan
        run.processed_columns = processed_columns
        run.save()
        return run
    except Exception as exc:  # noqa: BLE001
        logger.exception("Dataset analysis failed")
        run.status = "failed"
        run.error_message = str(exc)
        run.save(update_fields=["status", "error_message"])
        return run
