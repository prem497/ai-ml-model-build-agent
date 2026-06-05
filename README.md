# 🤖 Plain Text → Machine Learning Pipeline Agent

Convert natural language requests into **complete, executable ML pipelines** — automatically.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

> **No API key?** The system automatically falls back to rule-based pipeline planning — it still runs ML pipelines fully!

### 3. Start FastAPI Backend
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
📍 API: http://localhost:8000  
📍 Docs: http://localhost:8000/docs

### 4. Start Streamlit Frontend
```bash
streamlit run frontend/streamlit_app.py
```
📍 UI: http://localhost:8501

### 5. (Optional) Start MLflow UI
```bash
mlflow ui --port 5000
```
📍 MLflow: http://localhost:5000

---

## 💬 Example Prompts

```
Predict house prices using the California housing dataset with a Random Forest Regressor.
Do data preprocessing, train the model, evaluate it and show feature importance.
```
```
Classify iris flower species using a Decision Tree. Show accuracy and confusion matrix.
```
```
Detect breast cancer using XGBoost with 200 estimators. Show accuracy and feature importance.
```
```
Predict diabetes progression using Lasso regression and evaluate with R² and RMSE.
```

---

## 🏗️ Architecture

```
User Input (Plain Text)
        ↓
Streamlit Frontend  (localhost:8501)
        ↓
FastAPI Backend     (localhost:8000)
        ↓
LangChain Agent     (OpenAI / Gemini / Ollama)
        ↓
Pipeline Executor   (scikit-learn + XGBoost)
        ↓
MLflow Tracking     (localhost:5000)
        ↓
Storage             (SQLite + ChromaDB)
        ↓
Result Dashboard    (Code · Metrics · Charts · Steps)
```

## 📁 Project Structure

```
├── app/
│   ├── main.py               # FastAPI entry point
│   ├── api/
│   │   ├── routes.py         # API endpoints
│   │   └── schemas.py        # Pydantic models
│   ├── agent/
│   │   ├── llm_agent.py      # LangChain agent
│   │   ├── pipeline_planner.py
│   │   └── prompts.py
│   ├── engine/
│   │   ├── executor.py       # Pipeline orchestrator
│   │   ├── ml_steps.py       # ML functions
│   │   └── mlflow_tracker.py
│   ├── storage/
│   │   ├── db.py             # SQLite
│   │   └── vector_store.py   # ChromaDB
│   └── utils/
│       ├── code_gen.py       # Code generation
│       └── chart_gen.py      # Chart generation
├── frontend/
│   └── streamlit_app.py      # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```
