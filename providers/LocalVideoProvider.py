import os
import torch
import gc
import logging
from diffusers import AutoPipelineForText2Video
from diffusers.utils import export_to_video
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
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        # AutoPipelineForText2Video correctly resolves WanPipeline from model_index.json
        self.pipe = AutoPipelineForText2Video.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
        )
        # A100/H100/Blackwell have enough VRAM — run fully on GPU (skip cpu_offload for speed)
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
