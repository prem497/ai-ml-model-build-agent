"""
MLflow experiment tracking — logs params, metrics and model artifacts.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn


EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ML_Pipeline_Agent")


def _ensure_experiment() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    existing = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if existing is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)


def track_run(
    plan: Dict[str, Any],
    metrics: Dict[str, float],
    run_id: str,
    model: Optional[Any] = None,
) -> Optional[str]:
    """
    Log a pipeline run to MLflow.
    Returns the MLflow run_id string, or None on failure.
    """
    try:
        _ensure_experiment()

        with mlflow.start_run(run_name=f"pipeline_{run_id[:8]}") as run:
            # Tags
            mlflow.set_tags({
                "intent":     plan.get("intent", "unknown"),
                "model_type": plan.get("model", {}).get("type", "unknown"),
                "dataset":    plan.get("dataset", {}).get("source", "unknown"),
                "pipeline_id": run_id,
            })

            # Params
            model_params = plan.get("model", {}).get("params", {})
            for k, v in model_params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("test_size", plan.get("evaluation", {}).get("test_size", 0.2))

            # Metrics
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(name, float(value))

            # Log model artifact if provided
            if model is not None:
                try:
                    mlflow.sklearn.log_model(model, artifact_path="model")
                except Exception:
                    pass

            return run.info.run_id

    except Exception as e:
        # MLflow tracking is optional — don't crash the pipeline
        print(f"[MLflow] Tracking skipped: {e}")
        return None
