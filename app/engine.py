from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from PIL import Image
import math

PRESETS = {
    "push_in": "The camera smoothly dollies forward toward the subject.",
    "pull_back": "The camera smoothly dollies backward away from the subject.",
    "orbit_left": "The camera orbits counterclockwise around the stationary subject.",
    "pan_right": "The camera pans slowly toward the right.",
    "locked": "The camera is locked off on a tripod with no camera movement.",
}

@dataclass
class Shot:
    prompt: str
    preset: str
    seconds: int = 3
    fps: int = 8
    seed: int = 42

    def expanded_prompt(self) -> str:
        return f"{self.prompt.strip()} {PRESETS[self.preset]} Keep subject identity and appearance consistent."

class VideoEngine(Protocol):
    def generate(self, image_path: Path, output: Path, shot: Shot) -> None: ...

class MockEngine:
    """CPU-only motion preview. Not an AI video-generation model."""
    def generate(self, image_path: Path, output: Path, shot: Shot) -> None:
        import imageio.v2 as imageio
        import numpy as np
        source = Image.open(image_path).convert("RGB")
        source.thumbnail((768, 768))
        with imageio.get_writer(str(output), fps=shot.fps, codec="libx264", macro_block_size=16) as writer:
            total = shot.seconds * shot.fps
            for i in range(total):
                t = i / max(total - 1, 1)
                zoom = 1.0
                shift_x = 0
                if shot.preset == "push_in": zoom = 1 + .18 * t
                if shot.preset == "pull_back": zoom = 1.18 - .18 * t
                if shot.preset == "pan_right": shift_x = int(45 * t)
                if shot.preset == "orbit_left": shift_x = int(20 * math.sin(math.pi * t))
                w, h = int(440 * zoom), int(440 * zoom)
                frame = Image.new("RGB", (512, 512), (13, 17, 28))
                scaled = source.resize((w, h), Image.Resampling.LANCZOS)
                frame.paste(scaled, ((512-w)//2-shift_x, (512-h)//2))
                writer.append_data(np.asarray(frame))

class DiffusersSvdEngine:
    """Optional SVD adapter; camera presets do not condition this model."""
    def __init__(self):
        import torch
        from diffusers import StableVideoDiffusionPipeline
        self.torch = torch
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16, variant="fp16")
        self.pipe.enable_model_cpu_offload()

    def generate(self, image_path: Path, output: Path, shot: Shot) -> None:
        from diffusers.utils import export_to_video
        image = Image.open(image_path).convert("RGB").resize((1024, 576))
        generator = self.torch.manual_seed(shot.seed)
        frames = self.pipe(image, decode_chunk_size=2, generator=generator,
                           num_frames=25).frames[0]
        export_to_video(frames, str(output), fps=shot.fps)
