"""
Lumen — AI Video Enhancement Backend
=====================================

FastAPI server providing:
- Video upload with per-job settings (resolution, fps, AI upscaling, interpolation)
- Background processing queue (multi-user safe)
- Real-ESRGAN upscaling (auto-downloaded portable binary + models)
- RIFE frame interpolation (auto-downloaded portable binary + models)
- FFmpeg auto-download (Windows portable build) for decode/encode/mux
- GPU detection with automatic CPU fallback
- Secure file storage under ./storage/{uploads,outputs}
- Automatic deletion of files older than 24 hours

Run with:   python main.py
Bootstraps its own venv-less deps on first launch (see _ensure_deps).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
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
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------------------
# 0. Dependency bootstrap
# --------------------------------------------------------------------------------------

REQUIRED_PACKAGES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("python-multipart", "multipart"),
    ("requests", "requests"),
]


def _ensure_deps() -> None:
    """Install required Python packages if missing."""
    missing = []
    for pip_name, import_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[bootstrap] installing missing packages: {missing}", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--quiet", *missing]
        )


_ensure_deps()

import requests  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

# --------------------------------------------------------------------------------------
# 1. Paths & logging
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
STORAGE = ROOT / "storage"
UPLOADS = STORAGE / "uploads"
OUTPUTS = STORAGE / "outputs"
TOOLS = ROOT / "tools"
for p in (UPLOADS, OUTPUTS, TOOLS):
    p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lumen")

IS_WINDOWS = platform.system() == "Windows"

# --------------------------------------------------------------------------------------
# 2. Tool auto-download (FFmpeg / Real-ESRGAN / RIFE)
# --------------------------------------------------------------------------------------

FFMPEG_URL_WIN = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
REALESRGAN_URL_WIN = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
RIFE_URL_WIN = "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-windows.zip"

FFMPEG_URL_LINUX = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"


def _download(url: str, dest: Path) -> None:
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)


def _unzip(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(target)


def _find_exe(folder: Path, name_contains: str) -> Optional[Path]:
    suffix = ".exe" if IS_WINDOWS else ""
    for p in folder.rglob(f"*{suffix}"):
        if name_contains.lower() in p.name.lower() and p.is_file():
            return p
    return None


def ensure_ffmpeg() -> Path:
    """Return path to ffmpeg binary, downloading a portable build if needed."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return Path(on_path)

    ff_root = TOOLS / "ffmpeg"
    existing = _find_exe(ff_root, "ffmpeg") if ff_root.exists() else None
    if existing:
        return existing

    if not IS_WINDOWS:
        log.warning("ffmpeg not on PATH and auto-download only wired for Windows; install ffmpeg manually.")
        raise RuntimeError("ffmpeg not found")

    ff_root.mkdir(parents=True, exist_ok=True)
    archive = ff_root / "ffmpeg.zip"
    _download(FFMPEG_URL_WIN, archive)
    _unzip(archive, ff_root)
    archive.unlink(missing_ok=True)
    exe = _find_exe(ff_root, "ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg download succeeded but binary not found")
    return exe


def ensure_realesrgan() -> Optional[Path]:
    on_path = shutil.which("realesrgan-ncnn-vulkan")
    if on_path:
        return Path(on_path)
    re_root = TOOLS / "realesrgan"
    existing = _find_exe(re_root, "realesrgan") if re_root.exists() else None
    if existing:
        return existing
    if not IS_WINDOWS:
        log.warning("Real-ESRGAN auto-download only wired for Windows.")
        return None
    re_root.mkdir(parents=True, exist_ok=True)
    archive = re_root / "realesrgan.zip"
    try:
        _download(REALESRGAN_URL_WIN, archive)
        _unzip(archive, re_root)
        archive.unlink(missing_ok=True)
    except Exception as e:
        log.warning("Real-ESRGAN download failed: %s", e)
        return None
    return _find_exe(re_root, "realesrgan")


def ensure_rife() -> Optional[Path]:
    on_path = shutil.which("rife-ncnn-vulkan")
    if on_path:
        return Path(on_path)
    r_root = TOOLS / "rife"
    existing = _find_exe(r_root, "rife") if r_root.exists() else None
    if existing:
        return existing
    if not IS_WINDOWS:
        log.warning("RIFE auto-download only wired for Windows.")
        return None
    r_root.mkdir(parents=True, exist_ok=True)
    archive = r_root / "rife.zip"
    try:
        _download(RIFE_URL_WIN, archive)
        _unzip(archive, r_root)
        archive.unlink(missing_ok=True)
    except Exception as e:
        log.warning("RIFE download failed: %s", e)
        return None
    return _find_exe(r_root, "rife")


# --------------------------------------------------------------------------------------
# 3. GPU detection
# --------------------------------------------------------------------------------------

def has_gpu() -> bool:
    """Best-effort GPU detection — NVIDIA via nvidia-smi, Vulkan via vulkaninfo."""
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(["nvidia-smi"], check=True, capture_output=True, timeout=5)
            return True
        except Exception:
            pass
    if shutil.which("vulkaninfo"):
        return True
    return False


GPU_AVAILABLE = has_gpu()
log.info("GPU available: %s", GPU_AVAILABLE)

# --------------------------------------------------------------------------------------
# 4. Job model + queue
# --------------------------------------------------------------------------------------

RES_MAP = {"720p": (1280, 720), "1080p": (1920, 1080), "2K": (2560, 1440), "4K": (3840, 2160)}


@dataclass
class Job:
    id: str
    state: str = "queued"          # queued | uploading | processing | done | error
    progress: float = 0.0          # 0..100
    stage: str = "Queued"
    eta_seconds: Optional[int] = None
    download_url: Optional[str] = None
    error: Optional[str] = None
    settings: dict = field(default_factory=dict)
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "state": self.state,
            "progress": self.progress,
            "stage": self.stage,
            "etaSeconds": self.eta_seconds,
            "downloadUrl": self.download_url,
            "error": self.error,
        }


JOBS: dict[str, Job] = {}
JOB_QUEUE: "asyncio.Queue[str]" = asyncio.Queue()
JOBS_LOCK = threading.Lock()


# --------------------------------------------------------------------------------------
# 5. Processing pipeline
# --------------------------------------------------------------------------------------

def _run(cmd: list[str], on_line=None) -> int:
    log.info("$ %s", " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout
    for line in proc.stdout:
        line = line.rstrip()
        if on_line:
            on_line(line)
    proc.wait()
    return proc.returncode


def process_job(job: Job) -> None:
    """Full enhancement pipeline. Updates job.progress/stage in place."""
    try:
        settings = job.settings
        target_w, target_h = RES_MAP[settings["resolution"]]
        target_fps = int(settings["framerate"])
        do_upscale = bool(settings.get("upscale"))
        do_interp = bool(settings.get("interpolate"))

        ffmpeg = ensure_ffmpeg()
        realesrgan = ensure_realesrgan() if do_upscale else None
        rife = ensure_rife() if do_interp else None

        if do_upscale and not realesrgan:
            log.warning("Real-ESRGAN unavailable, falling back to FFmpeg scale.")
        if do_interp and not rife:
            log.warning("RIFE unavailable, falling back to FFmpeg minterpolate.")

        job.state = "processing"
        job.stage = "Decoding source"
        job.progress = 2

        workdir = Path(tempfile.mkdtemp(prefix=f"lumen_{job.id}_"))
        frames_in = workdir / "frames_in"
        frames_up = workdir / "frames_up"
        frames_interp = workdir / "frames_interp"
        for d in (frames_in, frames_up, frames_interp):
            d.mkdir()

        input_path = Path(job.input_path)
        # 1) Extract frames + audio
        audio_path = workdir / "audio.aac"
        _run([str(ffmpeg), "-y", "-i", str(input_path), "-vn", "-acodec", "copy", str(audio_path)])
        rc = _run(
            [str(ffmpeg), "-y", "-i", str(input_path), str(frames_in / "frame_%06d.png")],
        )
        if rc != 0:
            raise RuntimeError("ffmpeg frame extraction failed")
        job.progress = 20
        job.stage = "Frames extracted"

        # 2) Upscale (Real-ESRGAN) — optional
        if do_upscale and realesrgan:
            job.stage = "Real-ESRGAN upscaling"
            rc = _run([str(realesrgan), "-i", str(frames_in), "-o", str(frames_up), "-n", "realesrgan-x4plus"])
            if rc != 0:
                raise RuntimeError("Real-ESRGAN failed")
            source_dir = frames_up
        else:
            source_dir = frames_in
        job.progress = 55
        job.stage = "Upscale complete" if do_upscale else "Skipping upscale"

        # 3) Interpolate (RIFE) — optional
        if do_interp and rife:
            job.stage = "RIFE frame interpolation"
            rc = _run([str(rife), "-i", str(source_dir), "-o", str(frames_interp)])
            if rc != 0:
                raise RuntimeError("RIFE failed")
            final_frames = frames_interp
        else:
            final_frames = source_dir
        job.progress = 80
        job.stage = "Interpolation complete" if do_interp else "Skipping interpolation"

        # 4) Re-encode
        out_path = OUTPUTS / f"{job.id}.mp4"
        job.stage = f"Encoding {settings['resolution']} @ {settings['framerate']}fps"
        vf = f"scale={target_w}:{target_h}:flags=lanczos"
        if do_interp and not rife:
            vf = f"minterpolate=fps={target_fps},{vf}"
        encoder = ["-c:v", "h264_nvenc"] if GPU_AVAILABLE else ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
        cmd = [
            str(ffmpeg), "-y",
            "-framerate", str(target_fps),
            "-i", str(final_frames / "frame_%06d.png"),
        ]
        if audio_path.exists() and audio_path.stat().st_size > 0:
            cmd += ["-i", str(audio_path)]
        cmd += ["-vf", vf, "-r", str(target_fps), *encoder, "-pix_fmt", "yuv420p"]
        if audio_path.exists() and audio_path.stat().st_size > 0:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += [str(out_path)]
        rc = _run(cmd)
        if rc != 0:
            raise RuntimeError("ffmpeg encoding failed")

        job.output_path = str(out_path)
        job.download_url = f"/api/jobs/{job.id}/download"
        job.progress = 100
        job.stage = "Complete"
        job.state = "done"
        job.eta_seconds = 0
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception as e:
        log.exception("job %s failed", job.id)
        job.state = "error"
        job.error = str(e)


# --------------------------------------------------------------------------------------
# 6. FastAPI app
# --------------------------------------------------------------------------------------

app = FastAPI(title="Lumen — AI Video Enhancement")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "gpu": GPU_AVAILABLE, "platform": platform.system()}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), settings: str = Form("{}")):
    try:
        parsed = json.loads(settings)
    except json.JSONDecodeError:
        raise HTTPException(400, "invalid settings json")
    if parsed.get("resolution") not in RES_MAP:
        raise HTTPException(400, "invalid resolution")
    if str(parsed.get("framerate")) not in ("30", "60", "120"):
        raise HTTPException(400, "invalid framerate")

    job_id = uuid.uuid4().hex
    safe_name = f"{job_id}_{Path(file.filename or 'upload.mp4').name}"
    dest = UPLOADS / safe_name
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    job = Job(id=job_id, settings=parsed, input_path=str(dest))
    with JOBS_LOCK:
        JOBS[job_id] = job
    await JOB_QUEUE.put(job_id)
    log.info("queued job %s (%s)", job_id, file.filename)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.to_public()


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(404, "output not ready")
    return FileResponse(job.output_path, filename=f"lumen_{job_id}.mp4", media_type="video/mp4")


# --------------------------------------------------------------------------------------
# 7. Worker + cleanup
# --------------------------------------------------------------------------------------

async def worker_loop():
    log.info("worker started")
    while True:
        job_id = await JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if not job:
            continue
        await asyncio.get_event_loop().run_in_executor(None, process_job, job)


async def cleanup_loop():
    """Delete files older than 24h every 30 minutes."""
    while True:
        cutoff = time.time() - 24 * 3600
        for folder in (UPLOADS, OUTPUTS):
            for p in folder.iterdir():
                with contextlib.suppress(Exception):
                    if p.is_file() and p.stat().st_mtime < cutoff:
                        p.unlink()
                        log.info("deleted old file %s", p)
        # Also drop in-memory job records older than 24h
        with JOBS_LOCK:
            for jid in list(JOBS.keys()):
                if JOBS[jid].created_at < cutoff:
                    JOBS.pop(jid, None)
        await asyncio.sleep(1800)


@app.on_event("startup")
async def on_startup():
    # Pre-fetch ffmpeg so first job isn't slowed down (best-effort).
    try:
        ensure_ffmpeg()
    except Exception as e:
        log.warning("ffmpeg not pre-fetched: %s", e)
    asyncio.create_task(worker_loop())
    asyncio.create_task(cleanup_loop())
    log.info("Lumen backend ready · http://localhost:8000")


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
    

