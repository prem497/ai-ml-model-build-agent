"""
Reusable ML step functions — load, preprocess, split, train, evaluate.
"""
from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    load_iris,
    load_breast_cancer,
    load_wine,
    load_digits,
    load_diabetes,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)


# ─── Dataset Loaders ──────────────────────────────────────────────────────────

_SKLEARN_DATASETS = {
    "california_housing": fetch_california_housing,
    "iris":               load_iris,
    "breast_cancer":      load_breast_cancer,
    "wine":               load_wine,
    "digits":             load_digits,
    "diabetes":           load_diabetes,
}


def load_dataset(
    dataset_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Load a dataset from a sklearn built-in or a CSV URL.

    Returns
    -------
    X              : feature matrix (numpy ndarray)
    y              : target vector (numpy ndarray)
    feature_names  : list of feature name strings
    dataset_info   : metadata dict
    """
    source = dataset_config.get("source", "california_housing")

    if source in _SKLEARN_DATASETS:
        loader = _SKLEARN_DATASETS[source]
        if source == "california_housing":
            bunch = loader(as_frame=True)
        else:
            bunch = loader(as_frame=True)

        X = bunch.data.values.astype(float)
        y = bunch.target.values

        feature_names = list(bunch.feature_names) if hasattr(bunch, "feature_names") else [f"f{i}" for i in range(X.shape[1])]

        dataset_info = {
            "source":        source,
            "n_samples":     int(X.shape[0]),
            "n_features":    int(X.shape[1]),
            "feature_names": feature_names,
            "target_name":   dataset_config.get("target_column", "target"),
        }
        return X, y, feature_names, dataset_info

    if source == "csv_url":
        url = dataset_config.get("url", "")
        if not url:
            raise ValueError("dataset_url is required when source='csv_url'")
        df = pd.read_csv(url)
        target_col = dataset_config.get("target_column", df.columns[-1])
        X_df = df.drop(columns=[target_col])
        y = df[target_col].values

        # Encode object columns
        for col in X_df.select_dtypes(include="object").columns:
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

        feature_names = list(X_df.columns)
        X = X_df.values.astype(float)

        dataset_info = {
            "source":        url,
            "n_samples":     int(X.shape[0]),
            "n_features":    int(X.shape[1]),
            "feature_names": feature_names,
            "target_name":   target_col,
        }
        return X, y, feature_names, dataset_info

    raise ValueError(f"Unknown dataset source: {source}")


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_data(X: np.ndarray, steps: List[Dict[str, Any]]) -> np.ndarray:
    """
    Apply imputation and/or scaling to a numeric feature matrix.
    Returns the transformed matrix.
    """
    transformers = []

    for step in steps:
        kind = step.get("step")

        if kind == "imputation":
            strategy = step.get("strategy", "median")
            transformers.append(SimpleImputer(strategy=strategy))

        elif kind == "scaling":
            method = step.get("method", "standard")
            if method == "standard":
                transformers.append(StandardScaler())
            elif method == "minmax":
                transformers.append(MinMaxScaler())
            elif method == "robust":
                transformers.append(RobustScaler())

    if not transformers:
        # At minimum, impute NaNs
        transformers = [SimpleImputer(strategy="median")]

    # Chain transformers sequentially
    pipeline = Pipeline([(f"step_{i}", t) for i, t in enumerate(transformers)])
    return pipeline.fit_transform(X)


# ─── Train/Test Split ─────────────────────────────────────────────────────────

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ─── Model Factory ────────────────────────────────────────────────────────────

_REGRESSION_MODELS = {
    "random_forest":       lambda p: RandomForestRegressor(**p),
    "gradient_boosting":   lambda p: GradientBoostingRegressor(**p),
    "linear_regression":   lambda p: LinearRegression(),
    "ridge":               lambda p: Ridge(alpha=p.get("alpha", 1.0)),
    "lasso":               lambda p: Lasso(alpha=p.get("alpha", 0.1)),
    "svm":                 lambda p: SVR(kernel=p.get("kernel", "rbf")),
    "decision_tree":       lambda p: DecisionTreeRegressor(**{k: v for k, v in p.items() if k != "n_estimators"}),
    "knn":                 lambda p: KNeighborsRegressor(n_neighbors=p.get("n_neighbors", 5)),
    "xgboost":             lambda p: _make_xgb_regressor(p),
}

_CLASSIFICATION_MODELS = {
    "random_forest":       lambda p: RandomForestClassifier(**p),
    "gradient_boosting":   lambda p: GradientBoostingClassifier(**p),
    "logistic_regression": lambda p: LogisticRegression(max_iter=p.get("max_iter", 1000), random_state=p.get("random_state", 42)),
    "svm":                 lambda p: SVC(kernel=p.get("kernel", "rbf"), probability=True),
    "decision_tree":       lambda p: DecisionTreeClassifier(**{k: v for k, v in p.items() if k != "n_estimators"}),
    "knn":                 lambda p: KNeighborsClassifier(n_neighbors=p.get("n_neighbors", 5)),
    "xgboost":             lambda p: _make_xgb_classifier(p),
    "ridge":               lambda p: LogisticRegression(max_iter=1000),
    "lasso":               lambda p: LogisticRegression(max_iter=1000),
    "linear_regression":   lambda p: LogisticRegression(max_iter=1000),
}


def _make_xgb_regressor(params: Dict):
    try:
        from xgboost import XGBRegressor
        safe = {k: v for k, v in params.items() if k in ["n_estimators", "learning_rate", "max_depth", "random_state"]}
        return XGBRegressor(eval_metric="rmse", verbosity=0, **safe)
    except ImportError:
        return GradientBoostingRegressor(n_estimators=params.get("n_estimators", 200), random_state=42)


def _make_xgb_classifier(params: Dict):
    try:
        from xgboost import XGBClassifier
        safe = {k: v for k, v in params.items() if k in ["n_estimators", "learning_rate", "max_depth", "random_state"]}
        return XGBClassifier(eval_metric="logloss", verbosity=0, use_label_encoder=False, **safe)
    except ImportError:
        return GradientBoostingClassifier(n_estimators=params.get("n_estimators", 200), random_state=42)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_config: Dict[str, Any],
    intent: str = "regression",
) -> Any:
    """Instantiate and fit a model according to the pipeline plan."""
    model_type = model_config.get("type", "random_forest")
    params     = model_config.get("params", {"random_state": 42})

    if intent == "regression":
        factory = _REGRESSION_MODELS.get(model_type, _REGRESSION_MODELS["random_forest"])
    else:
        factory = _CLASSIFICATION_MODELS.get(model_type, _CLASSIFICATION_MODELS["random_forest"])

    model = factory(params)
    model.fit(X_train, y_train)
    return model


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    intent: str,
    requested_metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute requested evaluation metrics."""
    y_pred   = model.predict(X_test)
    metrics: Dict[str, float] = {}

    if intent == "regression":
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)
        metrics = {"r2": round(r2, 4), "mse": round(mse, 4), "rmse": round(rmse, 4)}

    elif intent == "classification":
        avg = "weighted"
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average=avg, zero_division=0)
        pr  = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rc  = recall_score(y_test, y_pred, average=avg, zero_division=0)
        metrics = {
            "accuracy":  round(float(acc), 4),
            "f1":        round(float(f1),  4),
            "precision": round(float(pr),  4),
            "recall":    round(float(rc),  4),
        }

    return metrics


# ─── Feature Importance ───────────────────────────────────────────────────────

def get_feature_importance(
    model: Any, feature_names: List[str]
) -> Optional[Dict[str, float]]:
    """Return feature importance dict (if the model supports it)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return dict(zip(feature_names, importances.tolist()))
    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_).ravel()
        return dict(zip(feature_names, coef.tolist()))
    return None
