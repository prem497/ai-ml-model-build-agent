"""
Parses raw LLM JSON output into validated pipeline plan structures.
"""
import json
import re
from typing import Dict, Any, List


# ─── Default / Fallback Plan ──────────────────────────────────────────────────

DEFAULT_PLAN: Dict[str, Any] = {
    "intent": "regression",
    "problem_description": "Predict median house values using California housing dataset",
    "dataset": {
        "source": "california_housing",
        "target_column": "MedHouseVal",
    },
    "preprocessing_steps": [
        {"step": "imputation", "strategy": "median"},
        {"step": "scaling",    "method":   "standard"},
    ],
    "model": {
        "type":   "random_forest",
        "params": {"n_estimators": 200, "random_state": 42},
    },
    "evaluation": {
        "metrics":   ["r2", "mse", "rmse"],
        "test_size": 0.2,
    },
    "pipeline_steps": [
        "Data Loading",
        "Data Preprocessing (Imputation + Scaling)",
        "Train/Test Split",
        "Model Training (Random Forest Regressor)",
        "Model Evaluation (MSE, R² Score)",
        "Feature Importance Visualization",
    ],
}


# ─── Parsers ──────────────────────────────────────────────────────────────────

def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Extract and validate JSON from an LLM response string.
    Falls back to the DEFAULT_PLAN if parsing fails.
    """
    # Strip potential markdown fences
    cleaned = re.sub(r"```(?:json)?", "", response_text, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip("`").strip()

    # Try the whole string first
    try:
        plan = json.loads(cleaned)
        return _validate_plan(plan)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find the first JSON object in the text
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            plan = json.loads(match.group())
            return _validate_plan(plan)
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback
    return DEFAULT_PLAN.copy()


def _validate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in missing fields with sensible defaults."""
    defaults = DEFAULT_PLAN.copy()

    plan.setdefault("intent",              defaults["intent"])
    plan.setdefault("problem_description", defaults["problem_description"])
    plan.setdefault("dataset",             defaults["dataset"])
    plan.setdefault("preprocessing_steps", defaults["preprocessing_steps"])
    plan.setdefault("model",               defaults["model"])
    plan.setdefault("evaluation",          defaults["evaluation"])
    plan.setdefault("pipeline_steps",      defaults["pipeline_steps"])

    plan["evaluation"].setdefault("test_size", 0.2)
    plan["model"].setdefault("params", {"random_state": 42})

    return plan


def plan_to_steps(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a pipeline plan's step names into structured step objects."""
    steps = []
    for i, step_name in enumerate(plan.get("pipeline_steps", []), start=1):
        steps.append(
            {
                "step_number": i,
                "name":        step_name,
                "description": f"Executing: {step_name}",
                "status":      "pending",
            }
        )
    return steps


def infer_intent_from_text(user_input: str) -> str:
    """
    Simple rule-based intent detection used as a fallback when LLM is unavailable.
    """
    text = user_input.lower()
    if any(w in text for w in ["classif", "detect", "identify", "predict class", "categor"]):
        return "classification"
    if any(w in text for w in ["cluster", "group", "segment", "kmeans", "k-means"]):
        return "clustering"
    return "regression"
