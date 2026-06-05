"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class PipelineRequest(BaseModel):
    user_input: str
    dataset_url: Optional[str] = None


class PipelineStepSchema(BaseModel):
    step_number: int
    name: str
    description: str
    status: str = "pending"  # pending | running | completed | failed


class PipelineResult(BaseModel):
    run_id: str
    user_input: str
    intent: str
    dataset_info: Dict[str, Any]
    pipeline_steps: List[PipelineStepSchema]
    generated_code: str
    metrics: Dict[str, Any]
    charts: Dict[str, str]          # key → base64-encoded PNG
    mlflow_run_id: Optional[str] = None
    execution_time: float
    created_at: datetime
    status: str


class PipelineHistoryItem(BaseModel):
    run_id: str
    user_input: str
    intent: str
    metrics: Dict[str, Any]
    status: str
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
