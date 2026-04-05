from __future__ import annotations

import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from environment.env import HealthcareAppointmentEnv
from environment.models import (
    SchedulerAction,
    SchedulerObservation,
    SchedulerState,
    StepResponse,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Healthcare Appointment Coordination Environment",
    description=(
        "OpenEnv-compatible environment for healthcare appointment scheduling. "
        "Exposes reset(), step(), and state() over HTTP."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

# ---------------------------------------------------------------------------
# Single in-memory env instance (one per container process)
# ---------------------------------------------------------------------------

_env = HealthcareAppointmentEnv()

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Body for POST /reset.  All fields are optional with sane defaults."""

    task_name: str = "easy"
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health() -> JSONResponse:
    """
    Health check endpoint required by Hugging Face Spaces.
    Always returns 200 while the server is alive.
    """
    return JSONResponse(content={"status": "ok", "service": "healthcare-appointment-coordination-env"})


@app.post("/reset", response_model=SchedulerObservation, tags=["env"])
def reset(body: ResetRequest) -> SchedulerObservation:
    """
    Start a new episode.

    - `task_name`: one of "easy", "medium", "hard" (default: "easy")
    - `seed`: integer seed for deterministic scenario generation (optional)

    Returns the initial SchedulerObservation (all patients pending, all slots available).
    """
    try:
        obs = _env.reset(task_name=body.task_name, seed=body.seed)
        return obs
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResponse, tags=["env"])
def step(action: SchedulerAction) -> StepResponse:
    """
    Submit one agent action and advance the environment.

    The request body must be a valid SchedulerAction.  Pydantic validates
    the schema (action_type, required fields per action type) before the
    action reaches the environment.

    Returns StepResponse with observation, reward, done flag, and diagnostics.

    Raises:
      400 – malformed action (Pydantic 422 for schema errors)
      409 – episode already done or reset() not yet called
    """
    try:
        result = _env.step(action)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=SchedulerState, tags=["env"])
def state() -> SchedulerState:
    """
    Return the full internal SchedulerState (deep copy).

    Intended for graders, evaluation tooling, and debugging.
    Not exposed to the agent during normal operation.

    Raises:
      409 – reset() not yet called
    """
    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )


# ---------------------------------------------------------------------------
# Startup / shutdown lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup() -> None:
    print("[server] Healthcare Appointment Coordination Environment is ready.")
    print(f"[server] Listening on port {int(os.environ.get('PORT', 7860))}")
    print("[server] Tasks available: easy, medium, hard")


# ---------------------------------------------------------------------------
# Runnable entry point for uvicorn / Docker / HF Spaces
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "environment.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        workers=1,          # single worker — env state is in-process memory
    )
