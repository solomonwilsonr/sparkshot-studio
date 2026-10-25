import os
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from .engine import PRESETS, Shot, MockEngine, DiffusersSvdEngine

ROOT = Path(os.environ.get("SPARKSHOT_DATA", "./outputs")).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="SparkShot Studio", version="0.1.0")
pool = ThreadPoolExecutor(max_workers=1)
lock = Lock()
jobs = {}
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = (DiffusersSvdEngine() if os.getenv("SPARKSHOT_ENGINE", "mock") == "svd"
                  else MockEngine())
    return engine

def run_job(job_id, image_path, shot):
    with lock:
        jobs[job_id]["status"] = "running"
    output = ROOT / job_id / "video.mp4"
    try:
        get_engine().generate(image_path, output, shot)
        with lock:
            jobs[job_id].update(status="done", video=f"/api/jobs/{job_id}/video")
    except Exception as exc:
        with lock:
            jobs[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}")

@app.get("/api/presets")
def presets():
    return PRESETS

@app.post("/api/jobs", status_code=202)
async def create_job(image: UploadFile = File(...), prompt: str = Form("Cinematic shot"),
                     preset: str = Form("push_in"), seconds: int = Form(3),
                     fps: int = Form(8), seed: int = Form(42)):
    if preset not in PRESETS:
        raise HTTPException(422, "Unknown camera preset")
    if not (1 <= seconds <= 5 and 4 <= fps <= 24):
        raise HTTPException(422, "seconds must be 1-5 and fps 4-24")
    if len(prompt) > 2000:
        raise HTTPException(422, "Prompt is too long")
    data = await image.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image too large (10MB maximum)")
    from PIL import Image, UnidentifiedImageError
    from io import BytesIO
    try:
        img = Image.open(BytesIO(data))
        img.verify()
        img = Image.open(BytesIO(data)).convert("RGB")
        if img.width * img.height > 20_000_000:
            raise ValueError("Image dimensions too large")
    except (UnidentifiedImageError, ValueError, OSError) as exc:
        raise HTTPException(422, f"Invalid image: {exc}") from exc
    job_id = uuid.uuid4().hex
    folder = ROOT / job_id
    folder.mkdir(mode=0o700)
    image_path = folder / "input.png"
    img.save(image_path)
    shot = Shot(prompt=prompt, preset=preset, seconds=seconds, fps=fps, seed=seed)
    with lock:
        jobs[job_id] = {"id": job_id, "status": "queued", "prompt": shot.expanded_prompt(), "preset": preset}
    pool.submit(run_job, job_id, image_path, shot)
    return jobs[job_id]

@app.get("/api/jobs/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]

@app.get("/api/jobs/{job_id}/video")
def video(job_id: str):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        raise HTTPException(404, "Video not ready")
    return FileResponse(ROOT / job_id / "video.mp4", media_type="video/mp4", filename="sparkshot.mp4")

@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "index.html").read_text()
