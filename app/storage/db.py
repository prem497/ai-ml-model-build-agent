"""
SQLite storage for pipeline run history using SQLAlchemy.
"""
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    create_engine, Column, String, Text, Float, DateTime, inspect
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pipeline_runs.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PipelineRunModel(Base):
    __tablename__ = "pipeline_runs"

    run_id         = Column(String, primary_key=True, index=True)
    user_input     = Column(Text, nullable=False)
    intent         = Column(String, default="unknown")
    pipeline_steps = Column(Text, default="[]")      # JSON
    generated_code = Column(Text, default="")
    metrics        = Column(Text, default="{}")       # JSON
    charts         = Column(Text, default="{}")       # JSON  (base64 strings — large)
    dataset_info   = Column(Text, default="{}")       # JSON
    mlflow_run_id  = Column(String, nullable=True)
    execution_time = Column(Float, default=0.0)
    status         = Column(String, default="completed")
    created_at     = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def save_pipeline_run(result: Dict[str, Any]):
    """Persist a completed pipeline run."""
    db = SessionLocal()
    try:
        run = PipelineRunModel(
            run_id         = result["run_id"],
            user_input     = result["user_input"],
            intent         = result.get("intent", "unknown"),
            pipeline_steps = json.dumps(result.get("pipeline_steps", [])),
            generated_code = result.get("generated_code", ""),
            metrics        = json.dumps(result.get("metrics", {})),
            charts         = json.dumps(result.get("charts", {})),
            dataset_info   = json.dumps(result.get("dataset_info", {})),
            mlflow_run_id  = result.get("mlflow_run_id"),
            execution_time = result.get("execution_time", 0.0),
            status         = result.get("status", "completed"),
            created_at     = result.get("created_at", datetime.utcnow()),
        )
        db.merge(run)
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def get_pipeline_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent pipeline runs (lightweight — no charts)."""
    db = SessionLocal()
    try:
        rows = (
            db.query(PipelineRunModel)
            .order_by(PipelineRunModel.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "run_id":     r.run_id,
                "user_input": r.user_input,
                "intent":     r.intent,
                "metrics":    json.loads(r.metrics),
                "status":     r.status,
                "created_at": r.created_at,
            }
            for r in rows
        ]
    finally:
        db.close()


def get_pipeline_by_id(run_id: str) -> Optional[Dict[str, Any]]:
    """Return a full pipeline run by ID, including charts."""
    db = SessionLocal()
    try:
        r = db.query(PipelineRunModel).filter(PipelineRunModel.run_id == run_id).first()
        if not r:
            return None
        return {
            "run_id":         r.run_id,
            "user_input":     r.user_input,
            "intent":         r.intent,
            "pipeline_steps": json.loads(r.pipeline_steps),
            "generated_code": r.generated_code,
            "metrics":        json.loads(r.metrics),
            "charts":         json.loads(r.charts),
            "dataset_info":   json.loads(r.dataset_info),
            "mlflow_run_id":  r.mlflow_run_id,
            "execution_time": r.execution_time,
            "status":         r.status,
            "created_at":     r.created_at,
        }
    finally:
        db.close()
