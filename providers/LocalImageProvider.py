import os
import torch
import gc
import logging
from diffusers import StableDiffusionXLPipeline
from monitoring.memory_watchdog import MemoryWatchdog

logger = logging.getLogger(__name__)

# Switched from gated FLUX.1-schnell to fully public SDXL-base-1.0
MODEL_PATH = os.path.join(os.environ.get("MODEL_DIR", "/app/models"), "stable-diffusion-xl-base-1.0")

class LocalImageProvider:
    def __init__(self):
        self.model_path = MODEL_PATH
        self.pipe = None

    def __enter__(self):
        logger.info(f"Loading SDXL from {self.model_path} into VRAM...")
        MemoryWatchdog.assert_vram_available(required_gb=10.0)
        try:
            # Prefer fp16 variant for smaller memory footprint
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
        except Exception:
            # Fallback: load without variant (e.g. if fp16 files weren't cached)
            logger.warning("SDXL fp16 variant not found, loading default weights...")
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            )
        # Offload to CPU between inference calls to keep VRAM free for other models
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Unloading SDXL and freeing VRAM...")
        if self.pipe:
            del self.pipe
            self.pipe = None
        MemoryWatchdog.enforce_cleanup()

    def generate(self, prompt: str, output_path: str, width: int = 832, height: int = 480) -> str:
        logger.info(f"Generating image for prompt: '{prompt[:60]}...'")

        image = self.pipe(
            prompt=prompt,
            guidance_scale=7.5,
            num_inference_steps=25,  # SDXL needs more steps than FLUX
            height=height,
            width=width
        ).images[0]

        image.save(output_path)
        logger.info(f"Saved image to {output_path}")
        return output_path
