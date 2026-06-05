"""
Python code generator — produces a readable, executable script for the pipeline plan.
"""
from __future__ import annotations

from typing import Any, Dict


# ─── Model Class Mapping ──────────────────────────────────────────────────────

_REGRESSION_CLASS = {
    "random_forest":       "RandomForestRegressor",
    "gradient_boosting":   "GradientBoostingRegressor",
    "linear_regression":   "LinearRegression",
    "ridge":               "Ridge",
    "lasso":               "Lasso",
    "svm":                 "SVR",
    "decision_tree":       "DecisionTreeRegressor",
    "knn":                 "KNeighborsRegressor",
    "xgboost":             "XGBRegressor",
}

_CLASSIFICATION_CLASS = {
    "random_forest":       "RandomForestClassifier",
    "gradient_boosting":   "GradientBoostingClassifier",
    "logistic_regression": "LogisticRegression",
    "svm":                 "SVC",
    "decision_tree":       "DecisionTreeClassifier",
    "knn":                 "KNeighborsClassifier",
    "xgboost":             "XGBClassifier",
    "ridge":               "LogisticRegression",
    "lasso":               "LogisticRegression",
    "linear_regression":   "LogisticRegression",
}

_DATASET_LOADERS = {
    "california_housing": "fetch_california_housing",
    "iris":               "load_iris",
    "breast_cancer":      "load_breast_cancer",
    "wine":               "load_wine",
    "digits":             "load_digits",
    "diabetes":           "load_diabetes",
}


def generate_pipeline_code(plan: Dict[str, Any]) -> str:
    """
    Generate a complete, executable Python script from a pipeline plan dict.
    """
    intent   = plan.get("intent", "regression")
    dataset  = plan.get("dataset", {})
    source   = dataset.get("source", "california_housing")
    model_cfg = plan.get("model", {})
    model_type = model_cfg.get("type", "random_forest")
    model_params = model_cfg.get("params", {"n_estimators": 200, "random_state": 42})
    preproc  = plan.get("preprocessing_steps", [])
    eval_cfg = plan.get("evaluation", {})
    test_size = eval_cfg.get("test_size", 0.2)

    if intent == "regression":
        model_class = _REGRESSION_CLASS.get(model_type, "RandomForestRegressor")
    else:
        model_class = _CLASSIFICATION_CLASS.get(model_type, "RandomForestClassifier")

    loader_fn = _DATASET_LOADERS.get(source, None)

    # Build import block
    imports = _build_imports(intent, model_type, source, preproc)

    # Dataset loading
    load_block = _build_load_block(source, dataset, loader_fn)

    # Preprocessing
    preproc_block = _build_preproc_block(preproc)

    # Split
    split_block = f"X_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size={test_size}, random_state=42\n)"

    # Train
    params_str = ", ".join(f"{k}={repr(v)}" for k, v in model_params.items())
    train_block = (
        f"model = {model_class}({params_str})\n"
        f"model.fit(X_train, y_train)\n"
        f"y_pred = model.predict(X_test)"
    )

    # Evaluate
    eval_block = _build_eval_block(intent)

    # Feature importance
    fi_block = _build_fi_block(model_type, intent)

    code = "\n\n".join(
        filter(
            None,
            [
                imports,
                "# ─── Load Dataset ────────────────────────────────────────────────",
                load_block,
                "# ─── Preprocessing ──────────────────────────────────────────────",
                preproc_block,
                "# ─── Train / Test Split ─────────────────────────────────────────",
                split_block,
                "# ─── Model Training ─────────────────────────────────────────────",
                train_block,
                "# ─── Evaluation ──────────────────────────────────────────────────",
                eval_block,
                fi_block,
            ],
        )
    )
    return code


# ─── Private Helpers ──────────────────────────────────────────────────────────

def _build_imports(intent, model_type, source, preproc) -> str:
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from sklearn.model_selection import train_test_split",
    ]

    # Metrics
    if intent == "regression":
        lines.append("from sklearn.metrics import mean_squared_error, r2_score")
    else:
        lines.append("from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix")

    # Dataset
    loader = _DATASET_LOADERS.get(source)
    if loader:
        lines.append(f"from sklearn.datasets import {loader}")
    else:
        lines.append("# CSV dataset — loaded via pandas")

    # Preprocessing
    has_imputer = any(p.get("step") == "imputation" for p in preproc)
    has_scaler  = any(p.get("step") == "scaling"    for p in preproc)
    if has_imputer:
        lines.append("from sklearn.impute import SimpleImputer")
    if has_scaler:
        scale_method = next((p.get("method") for p in preproc if p.get("step") == "scaling"), "standard")
        cls = {"standard": "StandardScaler", "minmax": "MinMaxScaler", "robust": "RobustScaler"}.get(scale_method, "StandardScaler")
        lines.append(f"from sklearn.preprocessing import {cls}")
    lines.append("from sklearn.pipeline import Pipeline")

    # Model
    if model_type == "xgboost":
        lines.append("from xgboost import XGBRegressor, XGBClassifier")
    elif intent == "regression":
        pkg = _regression_import(model_type)
        if pkg:
            lines.append(pkg)
    else:
        pkg = _classification_import(model_type)
        if pkg:
            lines.append(pkg)

    lines.append("import mlflow\nimport mlflow.sklearn")

    return "\n".join(lines)


def _regression_import(m: str) -> str:
    m = {
        "random_forest":     "from sklearn.ensemble import RandomForestRegressor",
        "gradient_boosting": "from sklearn.ensemble import GradientBoostingRegressor",
        "linear_regression": "from sklearn.linear_model import LinearRegression",
        "ridge":             "from sklearn.linear_model import Ridge",
        "lasso":             "from sklearn.linear_model import Lasso",
        "svm":               "from sklearn.svm import SVR",
        "decision_tree":     "from sklearn.tree import DecisionTreeRegressor",
        "knn":               "from sklearn.neighbors import KNeighborsRegressor",
    }.get(m, "")
    return m


def _classification_import(m: str) -> str:
    m = {
        "random_forest":       "from sklearn.ensemble import RandomForestClassifier",
        "gradient_boosting":   "from sklearn.ensemble import GradientBoostingClassifier",
        "logistic_regression": "from sklearn.linear_model import LogisticRegression",
        "svm":                 "from sklearn.svm import SVC",
        "decision_tree":       "from sklearn.tree import DecisionTreeClassifier",
        "knn":                 "from sklearn.neighbors import KNeighborsClassifier",
    }.get(m, "")
    return m


def _build_load_block(source, dataset, loader_fn) -> str:
    if loader_fn:
        target = dataset.get("target_column", "target")
        return (
            f"data = {loader_fn}(as_frame=True)\n"
            f"df   = data.frame\n"
            f"X    = data.data.values\n"
            f"y    = data.target.values\n"
            f"feature_names = list(data.feature_names)\n"
            f"print(f'Dataset shape: {{X.shape}}')"
        )
    url = dataset.get("url", "")
    target = dataset.get("target_column", "target")
    return (
        f"df = pd.read_csv('{url}')\n"
        f"X  = df.drop(columns=['{target}']).values\n"
        f"y  = df['{target}'].values\n"
        f"feature_names = list(df.drop(columns=['{target}']).columns)\n"
        f"print(f'Dataset shape: {{X.shape}}')"
    )


def _build_preproc_block(preproc) -> str:
    steps = []
    for p in preproc:
        kind = p.get("step")
        if kind == "imputation":
            steps.append(f"    ('imputer', SimpleImputer(strategy='{p.get('strategy', 'median')}'))")
        elif kind == "scaling":
            cls = {"standard": "StandardScaler", "minmax": "MinMaxScaler", "robust": "RobustScaler"}.get(p.get("method", "standard"), "StandardScaler")
            steps.append(f"    ('scaler', {cls}())")
    if not steps:
        steps.append("    ('imputer', SimpleImputer(strategy='median'))")

    return (
        "preprocessor = Pipeline([\n"
        + ",\n".join(steps)
        + "\n])\n"
        + "X = preprocessor.fit_transform(X)"
    )


def _build_eval_block(intent) -> str:
    if intent == "regression":
        return (
            "mse  = mean_squared_error(y_test, y_pred)\n"
            "rmse = np.sqrt(mse)\n"
            "r2   = r2_score(y_test, y_pred)\n"
            "print(f'MSE:  {mse:.4f}')\n"
            "print(f'RMSE: {rmse:.4f}')\n"
            "print(f'R²:   {r2:.4f}')\n\n"
            "# Actual vs Predicted plot\n"
            "plt.figure(figsize=(8, 6))\n"
            "plt.scatter(y_test, y_pred, alpha=0.5)\n"
            "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')\n"
            "plt.xlabel('Actual'); plt.ylabel('Predicted')\n"
            "plt.title('Actual vs Predicted')\n"
            "plt.tight_layout()\n"
            "plt.savefig('actual_vs_predicted.png', dpi=120)\n"
            "plt.show()"
        )
    return (
        "acc = accuracy_score(y_test, y_pred)\n"
        "f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)\n"
        "print(f'Accuracy: {acc:.4f}')\n"
        "print(f'F1 Score: {f1:.4f}')\n\n"
        "# Confusion Matrix\n"
        "cm = confusion_matrix(y_test, y_pred)\n"
        "plt.figure(figsize=(7, 5))\n"
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n"
        "plt.xlabel('Predicted'); plt.ylabel('True')\n"
        "plt.title('Confusion Matrix')\n"
        "plt.tight_layout()\n"
        "plt.savefig('confusion_matrix.png', dpi=120)\n"
        "plt.show()"
    )


def _build_fi_block(model_type: str, intent: str) -> str:
    tree_models = {"random_forest", "gradient_boosting", "decision_tree", "xgboost"}
    if model_type not in tree_models:
        return ""
    return (
        "# ─── Feature Importance ──────────────────────────────────────────────\n"
        "feature_importances = model.feature_importances_\n"
        "fi_series = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)\n"
        "plt.figure(figsize=(10, 6))\n"
        "sns.barplot(x=fi_series.values[:10], y=fi_series.index[:10], palette='viridis')\n"
        "plt.title('Top 10 Important Features')\n"
        "plt.xlabel('Importance')\n"
        "plt.tight_layout()\n"
        "plt.savefig('feature_importance.png', dpi=120)\n"
        "plt.show()"
    )
