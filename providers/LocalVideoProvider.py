import os
import logging
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

from monitoring.gpu_runtime import format_runtime_banner, get_preferred_torch_dtype
from monitoring.memory_watchdog import MemoryWatchdog

logger = logging.getLogger(__name__)
MODEL_PATH = os.path.join(os.environ.get("MODEL_DIR", "/app/models"), "Wan2.1-T2V-1.3B-Diffusers")


class LocalVideoProvider:
    def __init__(self):
        self.model_path = MODEL_PATH
        self.pipe = None

    def __enter__(self):
        logger.info(f"Loading {self.model_path} into VRAM...")
        MemoryWatchdog.assert_vram_available(required_gb=20.0)
        logger.info(format_runtime_banner())
        dtype = get_preferred_torch_dtype()
        self.pipe = WanPipeline.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
        )
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
        self.pipe.enable_vae_slicing()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(f"xformers not available, continuing without it: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Unloading Video Model and freeing VRAM...")
        if self.pipe:
            del self.pipe
            self.pipe = None
        MemoryWatchdog.enforce_cleanup()

    def generate_from_text(self, prompt: str, output_path: str, num_frames: int = 81, width: int = 832, height: int = 480) -> str:
        logger.info(f"Generating video for prompt: '{prompt[:50]}...'")
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=5.0,
                num_inference_steps=50,
            )
        frames = result.frames[0]
        export_to_video(frames, output_path, fps=16)
        logger.info(f"Saved video to {output_path}")
        return output_path
