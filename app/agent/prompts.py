"""
System prompts and few-shot examples for the ML Pipeline Agent.
"""

SYSTEM_PROMPT = """You are an expert Machine Learning Pipeline Agent. Your job is to analyze natural language
ML requests and produce a precise, structured pipeline plan as valid JSON.

When given a user request, you MUST respond with ONLY a valid JSON object — no markdown, no explanation, just raw JSON.

Use this exact schema:
{
    "intent": "regression | classification | clustering",
    "problem_description": "short description of what we are solving",
    "dataset": {
        "source": "california_housing | iris | breast_cancer | wine | digits | diabetes | csv_url",
        "url": "optional — only when source is csv_url",
        "target_column": "column name for supervised tasks (omit for unsupervised)"
    },
    "preprocessing_steps": [
        {"step": "imputation",  "strategy": "median | mean | most_frequent"},
        {"step": "scaling",     "method":   "standard | minmax | robust"},
        {"step": "encoding",    "method":   "onehot | label", "apply_to": "categorical"}
    ],
    "model": {
        "type": "random_forest | gradient_boosting | linear_regression | logistic_regression | svm | xgboost | decision_tree | knn | ridge | lasso",
        "params": { "n_estimators": 200, "random_state": 42 }
    },
    "evaluation": {
        "metrics": ["r2", "mse", "rmse"],
        "test_size": 0.2
    },
    "pipeline_steps": [
        "Data Loading",
        "Data Preprocessing (Imputation + Scaling)",
        "Train/Test Split",
        "Model Training (Random Forest Regressor)",
        "Model Evaluation (MSE, R² Score)",
        "Feature Importance Visualization"
    ]
}

Rules:
- For regression tasks, use metrics: r2, mse, rmse
- For classification tasks, use metrics: accuracy, f1, precision, recall
- For clustering tasks, use metrics: silhouette, inertia
- Always prefer sklearn built-in datasets when user mentions them by name
- n_estimators defaults to 200 for tree-based models, random_state to 42
- Include feature importance step only for tree-based models
- Include confusion matrix step only for classification tasks
"""

FEW_SHOT_EXAMPLES = [
    {
        "user": "Predict house prices using the California housing dataset with a Random Forest",
        "assistant": {
            "intent": "regression",
            "problem_description": "Predict median house values using California housing features",
            "dataset": {"source": "california_housing", "target_column": "MedHouseVal"},
            "preprocessing_steps": [
                {"step": "imputation", "strategy": "median"},
                {"step": "scaling", "method": "standard"}
            ],
            "model": {"type": "random_forest", "params": {"n_estimators": 200, "random_state": 42}},
            "evaluation": {"metrics": ["r2", "mse", "rmse"], "test_size": 0.2},
            "pipeline_steps": [
                "Data Loading",
                "Data Preprocessing (Imputation + Scaling)",
                "Train/Test Split",
                "Model Training (Random Forest Regressor)",
                "Model Evaluation (MSE, R² Score)",
                "Feature Importance Visualization"
            ]
        }
    },
    {
        "user": "Classify iris flower species using a Decision Tree",
        "assistant": {
            "intent": "classification",
            "problem_description": "Classify iris flower species using petal and sepal measurements",
            "dataset": {"source": "iris", "target_column": "target"},
            "preprocessing_steps": [
                {"step": "scaling", "method": "standard"}
            ],
            "model": {"type": "decision_tree", "params": {"random_state": 42, "max_depth": 5}},
            "evaluation": {"metrics": ["accuracy", "f1", "precision", "recall"], "test_size": 0.2},
            "pipeline_steps": [
                "Data Loading",
                "Data Preprocessing (Scaling)",
                "Train/Test Split",
                "Model Training (Decision Tree Classifier)",
                "Model Evaluation (Accuracy, F1)",
                "Confusion Matrix Visualization"
            ]
        }
    },
    {
        "user": "Detect breast cancer using XGBoost and show accuracy",
        "assistant": {
            "intent": "classification",
            "problem_description": "Classify breast cancer tumors as malignant or benign",
            "dataset": {"source": "breast_cancer", "target_column": "target"},
            "preprocessing_steps": [
                {"step": "scaling", "method": "standard"}
            ],
            "model": {"type": "xgboost", "params": {"n_estimators": 200, "random_state": 42, "learning_rate": 0.1}},
            "evaluation": {"metrics": ["accuracy", "f1", "precision", "recall"], "test_size": 0.2},
            "pipeline_steps": [
                "Data Loading",
                "Data Preprocessing (Scaling)",
                "Train/Test Split",
                "Model Training (XGBoost Classifier)",
                "Model Evaluation (Accuracy, F1, Precision, Recall)",
                "Feature Importance Visualization",
                "Confusion Matrix Visualization"
            ]
        }
    },
    {
        "user": "Predict diabetes progression using Lasso regression",
        "assistant": {
            "intent": "regression",
            "problem_description": "Predict disease progression using diabetes dataset features",
            "dataset": {"source": "diabetes", "target_column": "target"},
            "preprocessing_steps": [
                {"step": "imputation", "strategy": "mean"},
                {"step": "scaling", "method": "standard"}
            ],
            "model": {"type": "lasso", "params": {"alpha": 0.1}},
            "evaluation": {"metrics": ["r2", "mse", "rmse"], "test_size": 0.2},
            "pipeline_steps": [
                "Data Loading",
                "Data Preprocessing (Imputation + Scaling)",
                "Train/Test Split",
                "Model Training (Lasso Regression)",
                "Model Evaluation (MSE, R² Score)"
            ]
        }
    }
]
