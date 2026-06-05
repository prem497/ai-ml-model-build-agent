"""
FastAPI route handlers.
"""
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    HealthResponse,
    PipelineHistoryItem,
    PipelineRequest,
    PipelineResult,
)
from app.agent.llm_agent import run_pipeline_agent
from app.storage.db import get_pipeline_by_id, get_pipeline_history, save_pipeline_run

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
    )


@router.post("/pipeline/run", response_model=PipelineResult)
async def run_pipeline(request: PipelineRequest):
    run_id     = str(uuid.uuid4())
    start_time = time.time()

    try:
        result = run_pipeline_agent(
            user_input=request.user_input,
            dataset_url=request.dataset_url,
            run_id=run_id,
        )

        execution_time = time.time() - start_time

        full_result = {
            **result,
            "run_id":         run_id,
            "user_input":     request.user_input,
            "execution_time": round(execution_time, 3),
            "created_at":     datetime.utcnow(),
            "status":         "completed",
        }

        save_pipeline_run(full_result)

        return PipelineResult(**full_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/history", response_model=List[PipelineHistoryItem])
async def get_history():
    return get_pipeline_history()


@router.get("/pipeline/{run_id}", response_model=PipelineResult)
async def get_pipeline(run_id: str):
    run = get_pipeline_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return PipelineResult(**run)
