"""
FastAPI application entry point.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import mlflow
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.storage.db import init_db

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    init_db()
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "ML_Pipeline_Agent"))
    except Exception:
        pass
    print("[SUCCESS] ML Pipeline Agent API is ready.")
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────


app = FastAPI(
    title="ML Pipeline Agent API",
    description="Convert plain-text requests into end-to-end machine learning pipelines.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name":    "ML Pipeline Agent",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }
