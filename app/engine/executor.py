"""
Pipeline Execution Engine — orchestrates all ML steps end-to-end.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from app.engine.ml_steps import (
    load_dataset,
    preprocess_data,
    split_data,
    train_model,
    evaluate_model,
    get_feature_importance,
)
from app.utils.chart_gen import (
    generate_actual_vs_predicted_chart,
    generate_feature_importance_chart,
    generate_confusion_matrix_chart,
)


_TREE_MODELS = {
    "random_forest",
    "gradient_boosting",
    "decision_tree",
    "xgboost",
}


def execute_pipeline(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a full ML pipeline from the structured plan dict.

    Returns
    -------
    dict with keys:
        dataset_info  : metadata about the loaded dataset
        metrics       : evaluation metrics
        charts        : { chart_name: base64_png_string }
        model         : trained sklearn model object (for MLflow logging)
        y_test        : true labels (for chart generation)
        y_pred        : predictions
    """
    intent   = plan.get("intent", "regression")
    eval_cfg = plan.get("evaluation", {})
    test_size = eval_cfg.get("test_size", 0.2)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    X, y, feature_names, dataset_info = load_dataset(plan["dataset"])

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    X_proc = preprocess_data(X, plan.get("preprocessing_steps", []))

    # ── 3. Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X_proc, y, test_size)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    model = train_model(X_train, y_train, plan["model"], intent)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    metrics = evaluate_model(
        model, X_test, y_test, intent,
        eval_cfg.get("metrics", [])
    )

    # ── 6. Predict (for charts) ───────────────────────────────────────────────
    y_pred = model.predict(X_test)

    # ── 7. Charts ─────────────────────────────────────────────────────────────
    charts: Dict[str, str] = {}

    if intent == "regression":
        charts["actual_vs_predicted"] = generate_actual_vs_predicted_chart(y_test, y_pred)

    elif intent == "classification":
        charts["confusion_matrix"] = generate_confusion_matrix_chart(y_test, y_pred)

    # Feature importance (tree-based models)
    model_type = plan.get("model", {}).get("type", "")
    if model_type in _TREE_MODELS:
        fi = get_feature_importance(model, feature_names)
        if fi:
            charts["feature_importance"] = generate_feature_importance_chart(fi, feature_names)

    return {
        "dataset_info": dataset_info,
        "metrics":      metrics,
        "charts":       charts,
        "model":        model,
        "y_test":       y_test,
        "y_pred":       y_pred,
    }
