# pyre-ignore-all-errors
# -*- coding: utf-8 -*-
"""
Audition Service — Audio Analysis Microservice

Cloud Run microservice for audio analysis.
Single responsibility: audio bytes in → analysis JSON out.
"""

import asyncio
import os
import shutil
import tempfile
import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import httpx

from audition.services.audio_analysis import (
    analyze_audio_file,
    validate_audio_file,
)

FETCH_TIMEOUT = 120.0
MAX_AUDIO_SIZE = 500 * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("audition")


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Audition service is online.")
    logger.info("Audio analysis engine ready.")
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        logger.info("Semantic Extraction (Gemini) is ENABLED.")
    else:
        logger.warning("No Gemini key. Semantic Extraction DISABLED.")
    logger.info("Audio files will NOT be stored.")
    yield
    logger.info("Audition service shutting down.")


app = FastAPI(
    title="audition",
    description="Audio analysis with Gemini semantic extraction.",
    version="2.2.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_tracking(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    t0 = time.monotonic()
    logger.info(f"[{rid}] {request.method} {request.url.path}")
    response = await call_next(request)
    ms = int((time.monotonic() - t0) * 1000)
    logger.info(f"[{rid}] Completed in {ms}ms → {response.status_code}")
    response.headers["X-Request-ID"] = rid
    response.headers["X-Duration-Ms"] = str(ms)
    return response


@app.post("/internal/validate")
async def validate(file: UploadFile = File(...)) -> JSONResponse:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    try:
        result = await asyncio.to_thread(validate_audio_file, path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        if os.path.exists(path):
            os.remove(path)
    return JSONResponse(content={"status": "valid", "validation": result})


class AnalyzeRequest(BaseModel):
    local_path: str = Field(..., description="Absolute path on the mounted volume")


@app.post("/internal/analyze")
async def analyze(req: AnalyzeRequest) -> JSONResponse:
    path = req.local_path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found on volume.")

    # FUSE workaround: copy to local /tmp to avoid libsndfile SystemError on GCS FUSE mounts
    fd, local_tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    
    try:
        await asyncio.to_thread(shutil.copy2, path, local_tmp)
        
        try:
            await asyncio.to_thread(validate_audio_file, local_tmp)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        
        try:
            result = await asyncio.to_thread(analyze_audio_file, local_tmp)
        except Exception as e:
            logger.error(f"Analysis failed: {type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail="Analysis failed.")
            
        return JSONResponse(content=result)
    finally:
        if os.path.exists(local_tmp):
            os.remove(local_tmp)



class AnalyzeUrlRequest(BaseModel):
    audio_url: str = Field(..., description="Direct download URL")


@app.post("/internal/analyze-url")
async def analyze_url(req: AnalyzeUrlRequest) -> JSONResponse:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        path = tmp.name
    try:
        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT, follow_redirects=True, max_redirects=5,
        ) as client:
            async with client.stream("GET", req.audio_url) as resp:
                resp.raise_for_status()
                with open(path, "wb") as f:
                    async for chunk in resp.aiter_bytes(65536):
                        f.write(chunk)
    except httpx.HTTPStatusError as e:
        os.remove(path)
        raise HTTPException(status_code=502, detail=f"Fetch failed: {e.response.status_code}")
    except httpx.TimeoutException:
        os.remove(path)
        raise HTTPException(status_code=504, detail="Download timed out")
    except Exception as e:
        os.remove(path)
        raise HTTPException(status_code=502, detail=f"Fetch error: {e}")

    try:
        sz = os.path.getsize(path)
        if sz > MAX_AUDIO_SIZE:
            raise HTTPException(status_code=413, detail=f"Too large: {sz // 1048576}MB")
        if sz < 44:
            raise HTTPException(status_code=422, detail="File too small")
        try:
            await asyncio.to_thread(validate_audio_file, path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        try:
            result = await asyncio.to_thread(analyze_audio_file, path)
        except Exception as e:
            logger.error(f"Analysis failed: {type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail="Analysis failed.")
        return JSONResponse(content=result)
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.get("/")
async def index():
    return JSONResponse(
        content={
            "status": "online",
            "service": "audition",
            "engine": "Audio Analysis & Semantic Extraction Engine",
            "message": "Audition microservice is ready.",
            "documentation": "/docs"
        }
    )


@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ready", "service": "audition", "version": "2.2.0", "stores_audio": False})
