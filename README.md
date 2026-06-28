# Lumen Backend — AI Video Enhancement

FastAPI server that pairs with the Lumen frontend.

## What it does

- Accepts uploads with per-job settings (resolution / fps / upscale / interpolate)
- Runs a multi-user processing queue
- **Real-ESRGAN** upscaling (Vulkan, GPU-accelerated)
- **RIFE** frame interpolation (Vulkan, GPU-accelerated)
- **FFmpeg** for decode/encode/mux — uses `h264_nvenc` when an NVIDIA GPU is detected, otherwise `libx264`
- Auto-downloads portable builds of FFmpeg / Real-ESRGAN / RIFE on first run (Windows)
- Auto-installs missing Python deps on launch
- Deletes uploads & outputs older than 24 hours

## Run locally (Windows)

1. Install **Python 3.10+** — https://www.python.org/downloads/
2. Double-click `run.bat`
3. Open the Lumen frontend — it auto-detects the backend at `http://localhost:8000`

The first launch downloads ~300 MB of tools into `./tools/`. Subsequent launches are instant.

## Run locally (macOS / Linux)

```bash
chmod +x run.sh
./run.sh
```

On non-Windows you must install `ffmpeg`, `realesrgan-ncnn-vulkan`, and `rife-ncnn-vulkan` yourself (e.g. via Homebrew or your package manager). The backend will use whatever it finds on `PATH`.

## GPU

- NVIDIA GPU + recent driver → `nvidia-smi` is detected → `h264_nvenc` encoder
- Any Vulkan-capable GPU → Real-ESRGAN / RIFE run on GPU
- No GPU → automatic CPU fallback (`libx264`), Real-ESRGAN/RIFE still work but slower

## Endpoints

| Method | Path                              | Purpose                  |
|--------|-----------------------------------|--------------------------|
| GET    | `/health`                         | Health + GPU status      |
| POST   | `/api/jobs`                       | Upload + queue           |
| GET    | `/api/jobs/{id}`                  | Poll status              |
| GET    | `/api/jobs/{id}/download`         | Download finished MP4    |

## Storage

```
backend/
  storage/
    uploads/   ← source videos
    outputs/   ← enhanced MP4s
  tools/       ← auto-downloaded ffmpeg / realesrgan / rife
```

Files in `storage/` are deleted after 24 hours.

## Pointing the frontend at a different host

Open the deployed Lumen site, open DevTools console, and:

```js
localStorage.setItem("backend_url", "http://192.168.1.10:8000");
location.reload();
```
