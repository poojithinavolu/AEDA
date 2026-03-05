import logging

import numpy as np
import plotly.express as px

from analytics_engine.models import Visualization

logger = logging.getLogger("analytics_engine")


def _fallback_visualizations(frame):
    numeric_columns = [c for c in frame.columns if np.issubdtype(frame[c].dtype, np.number)]
    plan = []
    if numeric_columns:
        plan.append({"type": "histogram", "x": numeric_columns[0], "y": None})
    if len(numeric_columns) >= 2:
        plan.append({"type": "scatter", "x": numeric_columns[0], "y": numeric_columns[1]})
    return plan


def build_chart_html_from_plan(frame, plan: dict, include_plotlyjs="cdn"):
    chart_type = plan["type"]
    x_col = plan.get("x")
    y_col = plan.get("y")

    if chart_type == "scatter" and x_col and y_col:
        fig = px.scatter(frame, x=x_col, y=y_col, title=f"Scatter: {x_col} vs {y_col}")
    elif chart_type == "histogram" and x_col:
        fig = px.histogram(frame, x=x_col, title=f"Distribution of {x_col}")
    elif chart_type == "box" and x_col:
        fig = px.box(frame, y=x_col, title=f"Box Plot: {x_col}")
    elif chart_type == "bar" and x_col and y_col:
        fig = px.bar(frame, x=x_col, y=y_col, title=f"Bar: {x_col} vs {y_col}")
    elif chart_type == "line" and x_col and y_col:
        fig = px.line(frame, x=x_col, y=y_col, title=f"Line: {x_col} vs {y_col}")
    elif chart_type == "heatmap":
        corr = frame.select_dtypes(include="number").corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    else:
        raise ValueError(f"Unsupported chart plan: {plan}")
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def build_chart_html_list(frame, viz_plan: list[dict]) -> list[dict]:
    if not viz_plan:
        viz_plan = _fallback_visualizations(frame)

    html_items = []
    for idx, plan in enumerate(viz_plan, start=1):
        try:
            html = build_chart_html_from_plan(frame, plan, include_plotlyjs="cdn" if idx == 1 else False)
            html_items.append({"title": f"{plan['type'].title()} Chart {idx}", "plan": plan, "html": html})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Visualization generation failed for %s: %s", plan, exc)
    return html_items


def create_visualizations(dataset, analysis_run, owner, frame, viz_plan: list[dict]) -> list[Visualization]:
    if not viz_plan:
        viz_plan = _fallback_visualizations(frame)

    created = []
    for idx, plan in enumerate(viz_plan, start=1):
        chart_type = plan["type"]

        try:
            html = build_chart_html_from_plan(frame, plan, include_plotlyjs="cdn")
            viz = Visualization.objects.create(
                owner=owner,
                dataset=dataset,
                analysis_run=analysis_run,
                title=f"{chart_type.title()} Chart {idx}",
                chart_type=chart_type,
                spec_json=plan,
                html_snippet=html,
            )
            created.append(viz)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Visualization generation failed for %s: %s", plan, exc)

    return created
