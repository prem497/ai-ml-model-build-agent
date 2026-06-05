"""
LangChain-powered ML Pipeline Agent.
Supports OpenAI GPT-4, Google Gemini, and local Ollama.
Falls back to a rule-based plan when no API key is configured.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from app.agent.prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from app.agent.pipeline_planner import parse_llm_response, plan_to_steps, infer_intent_from_text
from app.engine.executor import execute_pipeline
from app.engine.mlflow_tracker import track_run
from app.storage.vector_store import store_pipeline_embedding, search_similar_pipelines
from app.utils.code_gen import generate_pipeline_code


# ─── LLM Factory ──────────────────────────────────────────────────────────────

def _build_llm():
    """Return the configured LLM, or None if no provider is available."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("your-"):
            return None
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                temperature=0,
                openai_api_key=api_key,
            )
        except Exception:
            return None

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key or api_key.startswith("your-"):
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=os.getenv("GOOGLE_MODEL", "gemini-1.5-pro"),
                google_api_key=api_key,
                temperature=0,
            )
        except Exception:
            return None

    if provider == "ollama":
        try:
            from langchain_community.llms import Ollama
            return Ollama(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "llama3"),
            )
        except Exception:
            return None

    return None


# ─── Intent via LLM ───────────────────────────────────────────────────────────

async def _call_llm(user_input: str) -> str:
    """
    Call the LLM with the system prompt and return the raw text response.
    Returns an empty string if LLM is unavailable.
    """
    llm = _build_llm()
    if llm is None:
        return ""

    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    # Build few-shot messages
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for ex in FEW_SHOT_EXAMPLES[:2]:          # include 2 examples max
        messages.append(HumanMessage(content=ex["user"]))
        messages.append(AIMessage(content=json.dumps(ex["assistant"])))
    messages.append(HumanMessage(content=user_input))

    try:
        response = await llm.ainvoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"[Agent] LLM call failed: {e}")
        return ""


# ─── Rule-based Fallback ──────────────────────────────────────────────────────

def _build_fallback_plan(user_input: str, dataset_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a sensible default plan based on keyword detection in the user input.
    Used when no LLM is available.
    """
    from app.agent.pipeline_planner import DEFAULT_PLAN
    import copy

    plan = copy.deepcopy(DEFAULT_PLAN)
    text = user_input.lower()

    # Infer intent
    plan["intent"] = infer_intent_from_text(user_input)

    # Infer dataset
    if "iris" in text:
        plan["dataset"] = {"source": "iris", "target_column": "target"}
    elif "breast cancer" in text or "cancer" in text:
        plan["dataset"] = {"source": "breast_cancer", "target_column": "target"}
    elif "wine" in text:
        plan["dataset"] = {"source": "wine", "target_column": "target"}
    elif "digit" in text or "mnist" in text:
        plan["dataset"] = {"source": "digits", "target_column": "target"}
    elif "diabetes" in text:
        plan["dataset"] = {"source": "diabetes", "target_column": "target"}
    elif dataset_url:
        plan["dataset"] = {"source": "csv_url", "url": dataset_url, "target_column": "target"}

    # Infer model
    model_map = {
        "random forest":       "random_forest",
        "gradient boosting":   "gradient_boosting",
        "xgboost":             "xgboost",
        "decision tree":       "decision_tree",
        "logistic":            "logistic_regression",
        "linear regression":   "linear_regression",
        "svm":                 "svm",
        "knn":                 "knn",
        "ridge":               "ridge",
        "lasso":               "lasso",
    }
    for keyword, mtype in model_map.items():
        if keyword in text:
            plan["model"]["type"] = mtype
            break

    # Adjust for intent
    if plan["intent"] == "classification":
        plan["evaluation"]["metrics"] = ["accuracy", "f1", "precision", "recall"]
        plan["pipeline_steps"] = [
            "Data Loading",
            "Data Preprocessing (Scaling)",
            "Train/Test Split",
            f"Model Training ({plan['model']['type'].replace('_', ' ').title()})",
            "Model Evaluation (Accuracy, F1)",
            "Confusion Matrix Visualization",
        ]
        if plan["model"]["type"] in {"random_forest", "gradient_boosting", "decision_tree", "xgboost"}:
            plan["pipeline_steps"].append("Feature Importance Visualization")

    plan["problem_description"] = f"Auto-generated pipeline based on: {user_input[:80]}"
    return plan


# ─── Main Agent Entry Point ───────────────────────────────────────────────────

async def run_pipeline_agent(
    user_input: str,
    dataset_url: Optional[str],
    run_id: str,
) -> Dict[str, Any]:
    """
    Orchestrate the full ML pipeline agent workflow:
    1. LLM intent understanding
    2. Pipeline planning
    3. ML execution
    4. Code generation
    5. MLflow tracking
    6. Vector store indexing
    """

    # ── Stage 1 & 2: Intent + Planning ────────────────────────────────────────
    llm_response = await _call_llm(user_input)

    if llm_response:
        plan = parse_llm_response(llm_response)
    else:
        # No LLM available — use rule-based fallback
        plan = _build_fallback_plan(user_input, dataset_url)

    if dataset_url:
        plan["dataset"]["url"]    = dataset_url
        plan["dataset"]["source"] = "csv_url"

    # ── Stage 3: Structured step list ─────────────────────────────────────────
    pipeline_steps = plan_to_steps(plan)

    # ── Stage 4: Execute ML pipeline ─────────────────────────────────────────
    execution_result = execute_pipeline(plan)

    # ── Stage 5: Generate Python code ─────────────────────────────────────────
    generated_code = generate_pipeline_code(plan)

    # ── Stage 6: MLflow tracking ──────────────────────────────────────────────
    mlflow_run_id = track_run(
        plan,
        execution_result["metrics"],
        run_id,
        model=execution_result.get("model"),
    )

    # ── Stage 7: Vector store ─────────────────────────────────────────────────
    store_pipeline_embedding(run_id, user_input, plan)

    # ── Compose result ────────────────────────────────────────────────────────
    return {
        "intent":         plan["intent"],
        "dataset_info":   execution_result["dataset_info"],
        "pipeline_steps": [{**s, "status": "completed"} for s in pipeline_steps],
        "generated_code": generated_code,
        "metrics":        execution_result["metrics"],
        "charts":         execution_result["charts"],
        "mlflow_run_id":  mlflow_run_id,
    }
