import json
import logging

from django.conf import settings

from analytics_engine.utils.validators import parse_json_response, validate_visualization_plan

logger = logging.getLogger("analytics_engine")


def _safe_llm_invoke(prompt: str) -> str:
    if not settings.GROQ_API_KEY:
        return ""

    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY, temperature=0)
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Copilot LLM invoke failed: %s", exc)
        return ""


def _fallback_answer(dataset_summary: dict, question: str) -> dict:
    columns = dataset_summary.get("columns", [])
    missing = dataset_summary.get("missing_values", {})
    quality = dataset_summary.get("quality_scorecard", {})
    numeric_cols = []
    for col, dtype in (dataset_summary.get("dtypes") or {}).items():
        lowered = str(dtype).lower()
        if "int" in lowered or "float" in lowered:
            numeric_cols.append(col)

    top_missing = sorted(missing.items(), key=lambda x: x[1], reverse=True)[:3]
    answer = (
        f"Quick summary for your question: {question}. "
        f"Dataset has {len(columns)} columns. "
        f"Overall quality score is {quality.get('overall_score', 'N/A')}. "
        f"Columns with highest missing counts: {top_missing}."
    )

    charts = []
    if len(numeric_cols) >= 2:
        charts.append({"type": "scatter", "x": numeric_cols[0], "y": numeric_cols[1]})
    if numeric_cols:
        charts.append({"type": "histogram", "x": numeric_cols[0], "y": None})

    return {"answer": answer, "charts": charts}


def ask_dataset_copilot(dataset_summary: dict, question: str, history: list[dict]) -> dict:
    prompt = (
        "You are AEDA Dataset Copilot. Answer user's analytics question using provided dataset summary only. "
        "Return only strict JSON with shape: "
        "{\"answer\":\"...\",\"charts\":[{\"type\":\"scatter|histogram|box|bar|line|heatmap\",\"x\":\"col\",\"y\":\"col_or_null\"}]}. "
        "Do not include markdown. Keep answer concise and actionable. "
        f"Question: {question}. "
        f"Dataset summary: {json.dumps(dataset_summary)}. "
        f"Recent conversation: {json.dumps(history[-6:])}."
    )

    raw = _safe_llm_invoke(prompt)
    parsed = parse_json_response(raw, {})
    if not parsed or not isinstance(parsed, dict):
        parsed = _fallback_answer(dataset_summary, question)

    answer = str(parsed.get("answer", "No answer generated.")).strip()
    raw_charts = parsed.get("charts", [])
    charts = validate_visualization_plan(raw_charts, set(dataset_summary.get("columns", [])))
    if not charts:
        charts = _fallback_answer(dataset_summary, question).get("charts", [])

    return {"answer": answer, "charts": charts}
