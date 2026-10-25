# SparkShot Studio

Independent DGX Spark-oriented video generation research prototype.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. Default backend uses a simple motion-preview renderer and does not generate new scenes with AI. Optional Stable Video Diffusion backend is experimental and requires compatible CUDA PyTorch, Diffusers, and model weights. Camera presets do not condition the AI model yet.

This project is independent and not affiliated with Higgsfield. Use only original or licensed inputs.
