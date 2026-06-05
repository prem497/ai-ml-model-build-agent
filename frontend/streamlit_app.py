"""
Streamlit Frontend — Plain Text → Machine Learning Pipeline Agent
Beautiful dark dashboard matching the reference design.
"""
from __future__ import annotations

import base64
import time
from io import BytesIO
from typing import Any, Dict, List, Optional
import os
import sys

# Add root directory to sys.path so 'app' module can be found
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import httpx
import streamlit as st

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Machine Model Build Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Auto-Start FastAPI Backend (For Streamlit Cloud) ───────────────────────
import subprocess
import socket
import sys
import os
import time

@st.cache_resource
def start_backend_once():
    def is_backend_running(port=8000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    if not is_backend_running(8000):
        print("Starting FastAPI backend automatically...")
        try:
            env = os.environ.copy()
            # Explicitly inject Streamlit secrets into the backend environment
            try:
                for k, v in st.secrets.items():
                    if isinstance(v, str):
                        env[k] = v
            except Exception:
                pass
            
            # Run detached and log to backend.log so we can debug
            log_file = open("backend.log", "w")
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            time.sleep(5)  # give it more time to boot on slow cloud VMs
        except Exception as e:
            print(f"Failed to start backend: {e}")

start_backend_once()

BACKEND_URL = "http://localhost:8000"

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Fira+Code:wght@400;500;600&display=swap');

/* ── Reset / Base ─────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background: #070b14 !important;
    color: #e2e8f0 !important;
}
[data-testid="stAppViewContainer"] > .main { background: transparent !important; }
[data-testid="block-container"] { padding-top: 1rem !important; }
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d !important;
}
/* Hide default streamlit header/footer */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* ── Hero Header ──────────────────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 2.2rem 2rem 1.8rem;
    background: linear-gradient(135deg,
        rgba(124,58,237,0.12) 0%,
        rgba(59,130,246,0.08) 50%,
        rgba(52,211,153,0.06) 100%);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 20px;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(124,58,237,0.15), transparent 60%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.3rem; font-weight: 900; margin: 0 0 0.4rem;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.2;
}
.hero p { color: #94a3b8; font-size: 1rem; margin: 0; }

/* ── Workflow Stage Badges ────────────────────────────────────────────────── */
.stages {
    display: flex; align-items: center; justify-content: center;
    flex-wrap: wrap; gap: 0.4rem; margin: 0.8rem 0 1.2rem;
}
.stage {
    padding: 0.35rem 0.9rem; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    border: 1px solid rgba(124,58,237,0.3);
    background: rgba(124,58,237,0.08); color: #a78bfa;
    transition: all 0.3s ease;
}
.stage.active {
    background: rgba(124,58,237,0.25); border-color: #a78bfa;
    color: #fff; box-shadow: 0 0 12px rgba(124,58,237,0.4);
    animation: pulse 1.2s infinite;
}
.stage.done {
    background: rgba(52,211,153,0.15); border-color: #34d399;
    color: #34d399;
}
.arrow { color: #374151; font-size: 0.9rem; }
@keyframes pulse {
    0%,100% { box-shadow: 0 0 12px rgba(124,58,237,0.4); }
    50%      { box-shadow: 0 0 24px rgba(124,58,237,0.7); }
}

/* ── Panel Cards ──────────────────────────────────────────────────────────── */
.panel {
    background: rgba(255,255,255,0.028);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
}
.panel-title {
    font-size: 0.92rem; font-weight: 700; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
    color: #e2e8f0; letter-spacing: 0.01em;
}

/* ── Metric Cards ─────────────────────────────────────────────────────────── */
.metric-row { display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 100px;
    background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(59,130,246,0.08));
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 12px; padding: 0.9rem 0.75rem; text-align: center;
}
.metric-label {
    font-size: 0.68rem; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.8rem; font-weight: 800; color: #34d399; line-height: 1;
}
.metric-good {
    display: inline-block; margin-top: 0.75rem;
    background: rgba(52,211,153,0.15); border: 1px solid #34d399;
    color: #34d399; padding: 0.2rem 0.75rem;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600;
}

/* ── Pipeline Steps ───────────────────────────────────────────────────────── */
.step-row {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.55rem 0.25rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.85rem; color: #cbd5e1;
}
.step-row:last-child { border-bottom: none; }
.step-check { color: #34d399; font-size: 1rem; flex-shrink: 0; }

/* ── Status Banner ────────────────────────────────────────────────────────── */
.status-banner {
    text-align: center; padding: 0.65rem 1.5rem;
    background: linear-gradient(90deg, rgba(124,58,237,0.15), rgba(52,211,153,0.12));
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 30px; margin: 0.75rem 0 1.25rem;
    font-size: 0.88rem; color: #e2e8f0; font-weight: 500;
}
.status-banner span { color: #a78bfa; font-weight: 700; }

/* ── Code Block Override ─────────────────────────────────────────────────── */
pre, code { font-family: 'Fira Code', monospace !important; font-size: 0.78rem !important; }

/* ── Sidebar History Items ────────────────────────────────────────────────── */
.hist-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 0.7rem 0.85rem; margin: 0.4rem 0;
    cursor: pointer; transition: all 0.2s;
}
.hist-card:hover {
    background: rgba(124,58,237,0.1);
    border-color: rgba(124,58,237,0.3);
}
.hist-tag {
    font-size: 0.65rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
    padding: 0.15rem 0.5rem; border-radius: 10px; margin-bottom: 0.3rem;
    display: inline-block;
}
.tag-regression     { background: rgba(124,58,237,0.2); color: #a78bfa; }
.tag-classification { background: rgba(59,130,246,0.2);  color: #60a5fa; }
.tag-clustering     { background: rgba(245,158,11,0.2);  color: #fbbf24; }
.hist-text { font-size: 0.8rem; color: #94a3b8; line-height: 1.4; }

/* ── Input / Chat ─────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background: #0d1117 !important; color: #e2e8f0 !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stChatMessage"] { background: rgba(255,255,255,0.02) !important; }

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton>button {
    background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    transition: opacity 0.2s !important;
}
.stButton>button:hover { opacity: 0.85 !important; }

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.02) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Session State Init ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "result" not in st.session_state:
    st.session_state.result = None
if "stages" not in st.session_state:
    st.session_state.stages = {k: "idle" for k in ["intent", "data", "plan", "code", "exec"]}
if "backend_url" not in st.session_state:
    st.session_state.backend_url = BACKEND_URL


# ─── Helpers ─────────────────────────────────────────────────────────────────

def b64_to_pil(b64: str):
    try:
        from PIL import Image
        data = base64.b64decode(b64)
        return Image.open(BytesIO(data))
    except Exception:
        return None


def tag_class(intent: str) -> str:
    return f"tag-{intent}" if intent in ("regression", "classification", "clustering") else "tag-regression"


def fmt_metric(v) -> str:
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def stage_class(key: str) -> str:
    s = st.session_state.stages.get(key, "idle")
    if s == "done":    return "stage done"
    if s == "active":  return "stage active"
    return "stage"


def set_stage(key: str, status: str):
    st.session_state.stages[key] = status


def reset_stages():
    for k in st.session_state.stages:
        st.session_state.stages[k] = "idle"


def fetch_history() -> List[Dict]:
    try:
        r = httpx.get(f"{st.session_state.backend_url}/api/pipeline/history", timeout=4)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def run_pipeline(user_input: str, dataset_url: Optional[str] = None) -> Optional[Dict]:
    url = f"{st.session_state.backend_url}/api/pipeline/run"
    payload = {"user_input": user_input}
    if dataset_url:
        payload["dataset_url"] = dataset_url
        
    try:
        # Import the backend logic directly to bypass unstable subprocess networking on Cloud
        import asyncio
        import uuid
        import time
        from datetime import datetime
        from app.agent.llm_agent import run_pipeline_agent
        from app.storage.db import save_pipeline_run

        run_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Streamlit is synchronous, so we run the async agent
        result = asyncio.run(run_pipeline_agent(
            user_input=user_input,
            dataset_url=dataset_url,
            run_id=run_id
        ))
        
        execution_time = time.time() - start_time
        full_result = {
            **result,
            "run_id":         run_id,
            "user_input":     user_input,
            "execution_time": round(execution_time, 3),
            "created_at":     datetime.utcnow(),
            "status":         "completed",
        }
        save_pipeline_run(full_result)
        return full_result
        
    except Exception as e:
        import traceback
        st.error("❌ Pipeline failed. Backend execution error:")
        st.code(traceback.format_exc(), language="text")
        return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Machine Model Build Agent")
    st.caption("Convert natural language into ML pipelines")
    st.divider()

    # Debug Log Viewer
    with st.expander("🛠️ Debug Backend Logs"):
        if os.path.exists("backend.log"):
            with open("backend.log", "r") as f:
                logs = f.read()
            st.code(logs[-2000:], language="text")
        else:
            st.write("No logs yet.")

    # Connection Status
    with st.expander("⚙️ Settings", expanded=False):
        new_url = st.text_input("Backend URL", value=st.session_state.backend_url)
        if new_url != st.session_state.backend_url:
            st.session_state.backend_url = new_url

        try:
            r = httpx.get(f"{st.session_state.backend_url}/api/health", timeout=3)
            if r.status_code == 200:
                info = r.json()
                st.success(f"✅ Connected  |  LLM: {info.get('llm_provider', '?')}")
            else:
                st.warning("⚠️ Backend returned error")
        except Exception:
            st.error("🔴 Backend offline")

        if st.button("📊 Launch MLflow UI"):
            st.code("mlflow ui --port 5000", language="bash")
            st.caption("Open http://localhost:5000 in your browser")

    st.divider()
    st.markdown("### 📚 Recent Pipelines")

    history = fetch_history()
    if history:
        for item in history[:8]:
            intent = item.get("intent", "regression")
            text   = item.get("user_input", "")[:55]
            metrics = item.get("metrics", {})
            m_str  = " · ".join(f"{k.upper()} {fmt_metric(v)}" for k, v in list(metrics.items())[:2])
            st.markdown(
                f"""<div class="hist-card">
                    <span class="hist-tag {tag_class(intent)}">{intent}</span>
                    <div class="hist-text">{text}…</div>
                    <div style="font-size:0.7rem;color:#6b7280;margin-top:0.25rem;">{m_str}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No runs yet — submit a prompt below!")

    st.divider()
    st.markdown(
        """<div style="font-size:0.72rem;color:#4b5563;text-align:center;line-height:1.6;">
        Powered by LangChain · FastAPI · MLflow · scikit-learn · ChromaDB
        </div>""",
        unsafe_allow_html=True,
    )


# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown(
    """<div class="hero">
        <h1>🤖 AI Machine Model Build Agent</h1>
        <p>Convert natural language requests into end-to-end machine learning pipelines</p>
    </div>""",
    unsafe_allow_html=True,
)

# ─── Workflow Stages Bar ───────────────────────────────────────────────────────
st.markdown(
    f"""<div class="stages">
        <div class="{stage_class('intent')}">1. Intent Understanding</div>
        <div class="arrow">→</div>
        <div class="{stage_class('data')}">2. Data Understanding</div>
        <div class="arrow">→</div>
        <div class="{stage_class('plan')}">3. Pipeline Planning</div>
        <div class="arrow">→</div>
        <div class="{stage_class('code')}">4. Code Generation</div>
        <div class="arrow">→</div>
        <div class="{stage_class('exec')}">5. Execution & Evaluation</div>
    </div>""",
    unsafe_allow_html=True,
)

# ─── Example Prompts ──────────────────────────────────────────────────────────
with st.expander("💡 Example Prompts — click to copy", expanded=False):
    examples = [
        "Predict house prices using the California housing dataset with a Random Forest Regressor. Do data preprocessing, train the model, evaluate it and show feature importance.",
        "Classify iris flower species using a Decision Tree. Show accuracy and confusion matrix.",
        "Detect breast cancer using XGBoost with 200 estimators. Show accuracy, F1 score and feature importance.",
        "Predict diabetes progression using Lasso regression and evaluate with R² and RMSE.",
        "Train a Gradient Boosting classifier on the wine dataset and show top 10 features.",
    ]
    for ex in examples:
        st.code(ex, language="text")

# ─── Result Dashboard ──────────────────────────────────────────────────────────
if st.session_state.result:
    res = st.session_state.result

    # Status banner
    metrics = res.get("metrics", {})
    m_str   = "  |  ".join(f"**{k.upper()}**: `{fmt_metric(v)}`" for k, v in list(metrics.items())[:3])
    st.markdown(
        f"""<div class="status-banner">
            ✨ <span>Your end-to-end ML pipeline is ready!</span>
            &nbsp;&nbsp;{m_str}
            &nbsp;&nbsp;You can run, modify or deploy this pipeline.
        </div>""",
        unsafe_allow_html=True,
    )

    # ── 2 × 2 Panel Grid ──────────────────────────────────────────────────────
    col_left, col_right = st.columns(2, gap="medium")

    # Panel A — Generated Python Code
    with col_left:
        st.markdown('<div class="panel-title">💻 A. Generated Python Code</div>', unsafe_allow_html=True)
        code = res.get("generated_code", "# No code generated")
        st.code(code, language="python", line_numbers=True)
        st.download_button(
            label="📥 Download pipeline.py",
            data=code,
            file_name="ml_pipeline.py",
            mime="text/x-python"
        )

    # Panel B — Model Performance
    with col_right:
        st.markdown('<div class="panel-title">📊 B. Model Performance</div>', unsafe_allow_html=True)

        # Metric cards
        metric_items = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float))]
        if metric_items:
            cards_html = '<div class="metric-row">'
            for name, val in metric_items[:4]:
                cards_html += f"""
                    <div class="metric-card">
                        <div class="metric-label">{name}</div>
                        <div class="metric-value">{fmt_metric(val)}</div>
                    </div>"""
            cards_html += "</div>"

            # Performance badge
            intent = res.get("intent", "regression")
            if intent == "regression":
                r2 = metrics.get("r2", 0)
                perf = "🟢 Good" if r2 > 0.7 else ("🟡 Fair" if r2 > 0.4 else "🔴 Low")
                cards_html += f'<div class="metric-good">Model Performance: {perf}</div>'
            else:
                acc = metrics.get("accuracy", 0)
                perf = "🟢 Good" if acc > 0.85 else ("🟡 Fair" if acc > 0.6 else "🔴 Low")
                cards_html += f'<div class="metric-good">Model Performance: {perf}</div>'

            st.markdown(cards_html, unsafe_allow_html=True)

        # Actual vs Predicted chart
        charts = res.get("charts", {})
        if "actual_vs_predicted" in charts:
            img = b64_to_pil(charts["actual_vs_predicted"])
            if img:
                st.image(img, caption="Actual vs Predicted", use_column_width=True)

        # Confusion matrix
        if "confusion_matrix" in charts:
            img = b64_to_pil(charts["confusion_matrix"])
            if img:
                st.image(img, caption="Confusion Matrix", use_column_width=True)

    col_left2, col_right2 = st.columns(2, gap="medium")

    # Panel C — Feature Importance
    with col_left2:
        st.markdown('<div class="panel-title">📈 C. Feature Importance</div>', unsafe_allow_html=True)
        if "feature_importance" in charts:
            img = b64_to_pil(charts["feature_importance"])
            if img:
                st.image(img, caption="Top 10 Important Features", use_column_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    # Panel D — Pipeline Steps
    with col_right2:
        st.markdown('<div class="panel-title">⚡ D. Generated Pipeline Steps</div>', unsafe_allow_html=True)

        step_icons = ["🗄️", "⚙️", "✂️", "🤖", "📊", "📈", "🎯", "🔍"]
        steps = res.get("pipeline_steps", [])
        steps_html = ""
        for i, step in enumerate(steps):
            name = step.get("name", step) if isinstance(step, dict) else str(step)
            icon = step_icons[i % len(step_icons)]
            steps_html += f"""
            <div class="step-row">
                <span class="step-check">✅</span>
                <span>{icon} {name}</span>
            </div>"""
        st.markdown(steps_html, unsafe_allow_html=True)

        # Dataset info
        dinfo = res.get("dataset_info", {})
        if dinfo:
            st.markdown("---")
            st.caption(
                f"📁 Dataset: `{dinfo.get('source','?')}` | "
                f"Samples: `{dinfo.get('n_samples','?')}` | "
                f"Features: `{dinfo.get('n_features','?')}`"
            )

        # Execution info
        exec_time = res.get("execution_time", 0)
        mlflow_id = res.get("mlflow_run_id")
        st.caption(f"⏱️ Execution time: `{exec_time:.2f}s`")
        if mlflow_id:
            st.caption(f"📊 MLflow run: `{mlflow_id[:16]}…`")

    st.divider()

# ─── Chat Interface ────────────────────────────────────────────────────────────
st.markdown("### 💬 Describe your ML pipeline")

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("e.g. Predict house prices using California housing with Random Forest..."):
    # Add to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process
    with st.chat_message("assistant"):
        stage_ph    = st.empty()
        progress_ph = st.empty()

        # Animate through stages
        stage_sequence = [
            ("intent", "🧠 Stage 1: Understanding your intent…"),
            ("data",   "📊 Stage 2: Analysing dataset…"),
            ("plan",   "📋 Stage 3: Planning pipeline steps…"),
            ("code",   "💻 Stage 4: Generating Python code…"),
            ("exec",   "🚀 Stage 5: Executing pipeline & evaluating…"),
        ]

        reset_stages()

        for i, (stage_key, stage_msg) in enumerate(stage_sequence[:-1]):
            set_stage(stage_key, "active")
            stage_ph.markdown(f"**{stage_msg}**")
            progress_ph.progress((i + 1) / len(stage_sequence))
            time.sleep(0.25)
            set_stage(stage_key, "done")

        set_stage("exec", "active")
        stage_ph.markdown("**🚀 Stage 5: Executing pipeline & evaluating…**")
        progress_ph.progress(0.9)

        # Call API
        result = run_pipeline(prompt)

        if result:
            # All stages done
            for k in st.session_state.stages:
                set_stage(k, "done")
            progress_ph.progress(1.0)

            metrics = result.get("metrics", {})
            m_str   = " · ".join(
                f"{k.upper()}: {fmt_metric(v)}"
                for k, v in list(metrics.items())[:3]
                if isinstance(v, (int, float))
            )
            stage_ph.markdown(f"✅ **Pipeline complete!** {m_str}")

            st.session_state.result = result
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"✅ **Pipeline complete!**\n\n"
                    f"📊 Metrics: {m_str}\n\n"
                    f"Scroll up to view the full results dashboard — "
                    f"generated code, performance charts, feature importance, and pipeline steps."
                ),
            })
        else:
            reset_stages()
            progress_ph.empty()
            stage_ph.markdown("❌ Pipeline failed — check the error above.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ Pipeline failed. Please check the backend logs.",
            })

    st.rerun()
