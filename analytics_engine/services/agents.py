import json
import logging
from typing import TypedDict

from django.conf import settings

from analytics_engine.utils.validators import parse_json_response

logger = logging.getLogger("analytics_engine")


class AgentState(TypedDict):
    dataset_context: dict
    feature_plan_text: str
    visualization_plan_text: str


def _infer_target_columns(columns: list[str]) -> set[str]:
    target_like = {"target", "label", "price", "y", "output"}
    inferred = set()
    for col in columns:
        lowered = col.lower()
        if lowered in target_like or lowered.endswith("_target") or lowered.endswith("_label"):
            inferred.add(col)
    return inferred


def _deterministic_feature_plan(dataset_context: dict) -> dict[str, list[str]]:
    columns = dataset_context.get("columns", [])
    dtypes = dataset_context.get("dtypes", {})
    missing = dataset_context.get("missing_values", {})
    inferred_targets = _infer_target_columns(columns)

    plan: dict[str, list[str]] = {}
    for col in columns:
        actions: list[str] = []
        dtype = str(dtypes.get(col, "")).lower()
        has_missing = float(missing.get(col, 0) or 0) > 0

        if has_missing:
            actions.append("handle_missing")

        if "object" in dtype or "category" in dtype:
            actions.append("one_hot_encode")
        elif ("int" in dtype or "float" in dtype) and col not in inferred_targets:
            actions.append("standardize")

        if actions:
            plan[col] = actions

    if not plan:
        numeric_non_target = [
            col for col in columns
            if ("int" in str(dtypes.get(col, "")).lower() or "float" in str(dtypes.get(col, "")).lower())
            and col not in inferred_targets
        ]
        for col in numeric_non_target[:3]:
            plan[col] = ["standardize"]
    return plan


def _deterministic_visualization_plan(dataset_context: dict) -> list[dict]:
    columns = dataset_context.get("columns", [])
    dtypes = dataset_context.get("dtypes", {})
    numeric_cols = [col for col in columns if "int" in str(dtypes.get(col, "")).lower() or "float" in str(dtypes.get(col, "")).lower()]

    plan: list[dict] = []
    if numeric_cols:
        plan.append({"type": "histogram", "x": numeric_cols[0], "y": None})
    if len(numeric_cols) >= 2:
        plan.append({"type": "scatter", "x": numeric_cols[0], "y": numeric_cols[-1]})
    if len(numeric_cols) >= 3:
        plan.append({"type": "heatmap", "x": None, "y": None})
    return plan


def _safe_llm_invoke(prompt: str) -> str:
    if not settings.GROQ_API_KEY:
        return ""

    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY, temperature=0)
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM invocation failed: %s", exc)
        return ""


def run_dataset_and_visualization_agents(dataset_context: dict) -> tuple[dict, list[dict]]:
    """
    LangGraph orchestration for two agents returning structured JSON strings.
    Falls back to deterministic defaults when LLM is unavailable.
    """

    def dataset_understanding_node(state: AgentState) -> AgentState:
        prompt = (
            "You are a data analyst. Return only JSON object mapping columns to actions. "
            "Allowed actions: normalize, standardize, handle_missing, one_hot_encode, drop. "
            f"Dataset context: {json.dumps(state['dataset_context'])}"
        )
        text = _safe_llm_invoke(prompt)
        state["feature_plan_text"] = text
        return state

    def visualization_planning_node(state: AgentState) -> AgentState:
        prompt = (
            "You are a visualization planner. Return only JSON array with objects "
            "{type, x, y}. Allowed types: scatter,histogram,box,bar,line,heatmap. "
            f"Dataset context: {json.dumps(state['dataset_context'])}. "
            f"Feature actions: {state.get('feature_plan_text', '{}')}"
        )
        text = _safe_llm_invoke(prompt)
        state["visualization_plan_text"] = text
        return state

    state: AgentState = {
        "dataset_context": dataset_context,
        "feature_plan_text": "{}",
        "visualization_plan_text": "[]",
    }

    try:
        from langgraph.graph import END, StateGraph

        graph = StateGraph(AgentState)
        graph.add_node("dataset_understanding", dataset_understanding_node)
        graph.add_node("visualization_planning", visualization_planning_node)
        graph.set_entry_point("dataset_understanding")
        graph.add_edge("dataset_understanding", "visualization_planning")
        graph.add_edge("visualization_planning", END)
        compiled = graph.compile()
        result = compiled.invoke(state)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LangGraph execution fallback: %s", exc)
        result = visualization_planning_node(dataset_understanding_node(state))

    feature_plan = parse_json_response(result.get("feature_plan_text", "{}"), {})
    visualization_plan = parse_json_response(result.get("visualization_plan_text", "[]"), [])

    if not feature_plan:
        logger.info("Using deterministic fallback feature plan")
        feature_plan = _deterministic_feature_plan(dataset_context)
    if not visualization_plan:
        logger.info("Using deterministic fallback visualization plan")
        visualization_plan = _deterministic_visualization_plan(dataset_context)
    return feature_plan, visualization_plan
