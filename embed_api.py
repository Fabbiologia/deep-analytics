"""
Embeddable API for the Deep Analytics chat agent.

This service wraps `run_analysis()` from `main.py` behind a small FastAPI app
and enables CORS for datamares.org. It is safe to run alongside your current
stack. No UI here—paired with `embed/embed.js` for a copy‑paste widget.

Run locally:
  uvicorn embed_api:app --host 0.0.0.0 --port 8080 --reload

ENV:
- ALLOW_ORIGINS: comma-separated list of allowed origins for CORS.
                 default: https://datamares.org,http://localhost:3000
"""
from __future__ import annotations

import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import the analysis function from your main agent file
try:
    from main import run_analysis, agent_available
except Exception as e:
    # Defer import-time failures into a clear API error
    run_analysis = None  # type: ignore
    agent_available = False  # type: ignore
    import traceback
    print("[embed_api] Warning: failed to import run_analysis from main.py")
    traceback.print_exc()

APP_TITLE = "Deep Analytics Chat API"
APP_DESCRIPTION = (
    "Simple JSON API that wraps the ecological monitoring agent's run_analysis()."
)
APP_VERSION = "0.1.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION, description=APP_DESCRIPTION)

# Configure CORS
_default_origins = [
    "https://datamares.org",
    "http://localhost:3000",
    "http://localhost:5173",
]
_env_origins = os.getenv("ALLOW_ORIGINS")
allow_origins: List[str] = (
    [o.strip() for o in _env_origins.split(",") if o.strip()] if _env_origins else _default_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

# Serve the embeddable assets at /static
_here = os.path.dirname(os.path.abspath(__file__))
_embed_dir = os.path.join(_here, "embed")
if os.path.isdir(_embed_dir):
    app.mount("/static", StaticFiles(directory=_embed_dir), name="static")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    elapsed_ms: int | None = None


@app.get("/health")
def health():
    return {
        "status": "ok" if agent_available else "agent_unavailable",
        "agent_available": bool(agent_available),
        "version": APP_VERSION,
        "allow_origins": allow_origins,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    import time

    if not agent_available or run_analysis is None:
        raise HTTPException(status_code=503, detail="Agent is not available. Check server logs and dependencies.")

    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message.")

    t0 = time.time()
    try:
        answer = run_analysis(msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}") from e
    dt_ms = int((time.time() - t0) * 1000)
    return ChatResponse(answer=answer, elapsed_ms=dt_ms)
