# pyre-ignore-all-errors
# -*- coding: utf-8 -*-
"""
Audition Service — Audio Analysis Microservice

Cloud Run microservice for audio analysis.
Single responsibility: audio bytes in → analysis JSON out.

This service does NOT:
  - Store audio files
  - Keep state between requests (stateless)
  - Compress or degrade the input
  - Make mastering decisions (that is deliberation's domain)

This service DOES:
  - Validate audio integrity
  - Compute 9-dimensional Time-Series Circuit Envelopes
  - Return physical engineering coordinates
  - Forget everything after the response is sent
"""

import os
import tempfile
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import httpx

from audition.services.audio_analysis import analyze_audio_file, validate_audio_file

FETCH_TIMEOUT = 120.0
MAX_AUDIO_SIZE = 500 * 1024 * 1024  # 500MB

# ──────────────────────────────────────────
# Logging
# ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("audition")


# ──────────────────────────────────────────
# Application Lifecycle
# ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Startup: Verify that the engine is operational.
    Shutdown: Nothing to clean up. We hold no state. We store no files.
    """
    logger.info("Audition service is online.")
    logger.info("Audio analysis engine ready.")
    logger.info("Audio files will NOT be stored.")
    yield
    logger.info("Audition service shutting down.")


# ──────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────
app = FastAPI(
    title="audition",
    description="Time-Series Circuit Envelope Function — Converts the flow of musical time into 9-dimensional engineering coordinates.",
    version="2.1.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────
# Middleware: Request Tracking
# ──────────────────────────────────────────
@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    """
    Assigns a unique request_id for traceability.
    Logs duration. Does NOT log audio content.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.monotonic()

    logger.info(f"[{request_id}] {request.method} {request.url.path}")

    response = await call_next(request)

    duration_ms = int((time.monotonic() - start_time) * 1000)
    logger.info(f"[{request_id}] Completed in {duration_ms}ms → {response.status_code}")

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Duration-Ms"] = str(duration_ms)

    return response


# ══════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════
@app.post("/internal/validate")
async def validate(file: UploadFile = File(...)) -> JSONResponse:
    """
    Validates audio file integrity without running full analysis.
    Streams input to temporary disk, evaluates, then deletes file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
        input_path = tmp_in.name
        while True:
            chunk = await file.read(65536)
            if not chunk:
                break
            tmp_in.write(chunk)

    try:
        validation_result = validate_audio_file(input_path)
    except ValueError as validation_error:
        os.remove(input_path)
        raise HTTPException(status_code=422, detail=str(validation_error))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    return JSONResponse(content={
        "status": "valid",
        "validation": validation_result,
    })


@app.post("/internal/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    """
    Core endpoint: Converts raw audio into a Time-Series Circuit Envelope.
    Streams audio to local disk, runs analysis, removes file immediately.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
        input_path = tmp_in.name
        while True:
            chunk = await file.read(65536)
            if not chunk:
                break
            tmp_in.write(chunk)

    # Step 1: Validate
    try:
        validate_audio_file(input_path)
    except ValueError as validation_error:
        os.remove(input_path)
        raise HTTPException(status_code=422, detail=str(validation_error))

    # Step 2: Analyze (Time-Series Circuit Envelope generation)
    try:
        analysis_result = analyze_audio_file(input_path)
    except Exception as analysis_error:
        logger.error(f"Analysis failed: {type(analysis_error).__name__}: {analysis_error}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {type(analysis_error).__name__}: {analysis_error}",
        )
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    return JSONResponse(content=analysis_result)


class AnalyzeUrlRequest(BaseModel):
    audio_url: str = Field(..., description="Direct download URL for audio file")


@app.post("/internal/analyze-url")
async def analyze_url(req: AnalyzeUrlRequest) -> JSONResponse:
    """
    URL-based analysis: streams audio from URL to local disk, runs full analysis, deletes file.
    Completely avoids OOM by streaming to temporary files during download.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
        input_path = tmp_in.name

    try:
        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT, follow_redirects=True, max_redirects=5,
        ) as client:
            async with client.stream("GET", req.audio_url) as resp:
                resp.raise_for_status()
                with open(input_path, "wb") as f_in:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        f_in.write(chunk)
    except httpx.HTTPStatusError as e:
        os.remove(input_path)
        raise HTTPException(status_code=502, detail=f"Failed to fetch audio: {e.response.status_code}")
    except httpx.TimeoutException:
        os.remove(input_path)
        raise HTTPException(status_code=504, detail="Audio download timed out")
    except Exception as e:
        os.remove(input_path)
        raise HTTPException(status_code=502, detail=f"Audio fetch error: {type(e).__name__}: {e}")

    try:
        file_size = os.path.getsize(input_path)
        if file_size > MAX_AUDIO_SIZE:
            raise HTTPException(status_code=413, detail=f"Audio too large: {file_size / 1024 / 1024:.0f}MB (max 500MB)")

        if file_size < 44:
            raise HTTPException(status_code=422, detail="Downloaded file too small to be valid audio")

        try:
            validate_audio_file(input_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        try:
            analysis_result = analyze_audio_file(input_path)
        except Exception as e:
            logger.error(f"Analysis failed: {type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {type(e).__name__}: {e}")

        return JSONResponse(content=analysis_result)
        
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@app.get("/")
async def index() -> JSONResponse:
    """Root endpoint providing service identity."""
    return JSONResponse(content={
        "status": "online",
        "service": "audition",
        "engine": "Time-Series Circuit Envelope Function",
        "message": "AI-driven professional audio analysis microservice is ready.",
        "documentation": "/docs"
    })


@app.get("/health")
async def health() -> JSONResponse:
    """
    Health check for Cloud Run.
    Returns service status.
    """
    return JSONResponse(content={
        "status": "ready",
        "service": "audition",
        "version": "2.1.0",
        "engine": "audio_analysis",
        "envelope_dimensions": 8,
        "stores_audio": False,
    })


# ══════════════════════════════════════════
# Standalone Execution (Local Development)
# ══════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting audition service in local development mode.")
    logger.info("Endpoints:")
    logger.info("  POST /internal/validate  — Validate audio integrity")
    logger.info("  POST /internal/analyze   — Full Time-Series Circuit Envelope analysis")
    logger.info("  GET  /health             — Health check")

    uvicorn.run(
        "audition.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
