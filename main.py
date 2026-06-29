"""
Lumen — AI Video Enhancement Backend (FULL FIXED VERSION)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ------------------------------------------------------
# Logging
# ------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lumen")

# ------------------------------------------------------
# Paths
# ------------------------------------------------------

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
TOOLS = ROOT / "tools"

for p in (UPLOADS, OUTPUTS, TOOLS):
    p.mkdir(parents=True, exist_ok=True)

IS_WINDOWS = platform.system() == "Windows"

# ------------------------------------------------------
# Dependencies (optional bootstrap)
# ------------------------------------------------------

REQUIRED_PACKAGES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("python-multipart", "multipart"),
    ("requests", "requests"),
]

def ensure_deps():
    missing = []
    for pip_name, import_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        log.info(f"Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])

ensure_deps()

import requests

# ------------------------------------------------------
# FFmpeg
# ------------------------------------------------------

def get_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("FFmpeg not found in system PATH")
    return ff

# ------------------------------------------------------
# Job model
# ------------------------------------------------------

@dataclass
class Job:
    id: str
    state: str = "queued"
    progress: int = 0
    stage: str = "queued"
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)

JOBS: dict[str, Job] = {}
QUEUE: asyncio.Queue[str] = asyncio.Queue()
LOCK = threading.Lock()

# ------------------------------------------------------
# FFmpeg runner (IMPORTANT FIXED)
# ------------------------------------------------------

def run(cmd: list[str]):
    log.info("RUN: %s", " ".join(cmd))

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    output = []
    assert p.stdout

    for line in p.stdout:
        line = line.strip()
        output.append(line)
        log.info(line)

    p.wait()

    if p.returncode != 0:
        raise RuntimeError("FFmpeg error:\n" + "\n".join(output[-30:]))

# ------------------------------------------------------
# Processing pipeline (FULL FIXED)
# ------------------------------------------------------

def process(job: Job):
    try:
        ffmpeg = get_ffmpeg()

        job.state = "processing"
        job.progress = 10
        job.stage = "extracting frames"

        workdir = Path(tempfile.mkdtemp())
        frames = workdir / "frames"
        frames.mkdir()

        input_path = Path(job.input_path)

        # 1. Extract frames (FIXED)
        run([
            ffmpeg,
            "-y",
            "-i", str(input_path),
            str(frames / "frame_%06d.png")
        ])

        job.progress = 40
        job.stage = "encoding frames"

        # 2. Rebuild video (FIXED)
        output_file = OUTPUTS / f"{job.id}.mp4"

        run([
            ffmpeg,
            "-y",
            "-framerate", "30",
            "-i", str(frames / "frame_%06d.png"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_file)
        ])

        job.output_path = str(output_file)
        job.state = "done"
        job.progress = 100
        job.stage = "complete"

    except Exception as e:
        job.state = "error"
        job.error = str(e)
        log.exception("JOB FAILED")

# ------------------------------------------------------
# Worker
# ------------------------------------------------------

async def worker():
    log.info("worker started")

    while True:
        job_id = await QUEUE.get()
        job = JOBS.get(job_id)

        if job:
            await asyncio.get_event_loop().run_in_executor(None, process, job)

# ------------------------------------------------------
# FastAPI app
# ------------------------------------------------------

app = FastAPI(title="Lumen AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# Upload
# ------------------------------------------------------

@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    settings: str = Form("{}")
):
    try:
        settings_data = json.loads(settings)
    except:
        raise HTTPException(400, "Invalid JSON")

    job_id = uuid.uuid4().hex
    input_path = UPLOADS / f"{job_id}.mp4"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    job = Job(
        id=job_id,
        input_path=str(input_path)
    )

    JOBS[job_id] = job
    await QUEUE.put(job_id)

    return {"id": job_id}

# ------------------------------------------------------
# Status
# ------------------------------------------------------

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    return job.__dict__

# ------------------------------------------------------
# Download
# ------------------------------------------------------

@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    job = JOBS.get(job_id)

    if not job or not job.output_path:
        raise HTTPException(404, "No output yet")

    return FileResponse(job.output_path, media_type="video/mp4")

# ------------------------------------------------------
# Startup
# ------------------------------------------------------

@app.on_event("startup")
async def startup():
    asyncio.create_task(worker())
    log.info("Lumen backend ready")

# ------------------------------------------------------
# Run
# ------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
