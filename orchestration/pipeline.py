import os
import json
import logging
import re
import gc
import torch
from typing import Dict, Any

from storage.checkpoint import CheckpointManager
from providers.LocalChatProvider import LocalChatProvider
from providers.LocalImageProvider import LocalImageProvider
from providers.LocalVideoProvider import LocalVideoProvider
from pipelines.ffmpeg_assembly import FFmpegAssemblyPipeline

logger = logging.getLogger(__name__)


def cleanup_vram():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


class CinematicPipeline:
    def __init__(self, output_dir="/app/outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint = CheckpointManager()

    def run(self, job_id: str, prompt: str):
        job_dir = os.path.join(self.output_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)

        logger.info(f"[{job_id}] Starting/Resuming pipeline for prompt: {prompt}")
        state = self.checkpoint.load_state(job_id)

        # Guard: return immediately if already completed with valid video on disk
        early = self._assert_not_completed(state, job_id)
        if early:
            return early

        if not state.get("prompt"):
            state["prompt"] = prompt
            state["status"] = "generating_screenplay"
            self.checkpoint.save_state(job_id, state)

        if state["status"] == "generating_screenplay":
            logger.info(f"[{job_id}] Generating Screenplay...")
            with LocalChatProvider() as chat:
                system_prompt = (
                    "You are a cinematic AI director. "
                    "Given a story prompt, split it into 3 consecutive short scenes. "
                    "Maintain the exact same protagonist character description across all scenes to ensure visual consistency. "
                    "Output strictly valid JSON in this format: "
                    '{"scenes": [{"scene_number": 1, "image_prompt": "Detailed visual description for an image generator", "video_prompt": "Detailed motion description for a video generator"}, ...]}'
                )
                response = chat.generate(prompt, system_prompt=system_prompt)

            cleanup_vram()

            try:
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if not match:
                    raise ValueError(f"No JSON block found in response")
                json_str = match.group(0)
                screenplay = json.loads(json_str)

                if "scenes" not in screenplay or not isinstance(screenplay["scenes"], list) or len(screenplay["scenes"]) == 0:
                    raise ValueError("JSON parsed successfully but 'scenes' array is missing or empty.")

            except Exception as e:
                logger.error(f"[{job_id}] Screenplay parsing failed ({e}). Falling back to generic 1-scene screenplay.")
                screenplay = {
                    "scenes": [
                        {
                            "scene_number": 1,
                            "image_prompt": f"High quality cinematic shot of: {prompt}",
                            "video_prompt": f"Slow cinematic pan, highly detailed: {prompt}",
                        }
                    ]
                }

            with open(os.path.join(job_dir, "screenplay.json"), "w") as f:
                json.dump(screenplay, f, indent=4)

            state["screenplay"] = screenplay
            state["status"] = "generating_images"
            self.checkpoint.save_state(job_id, state)

        if state["status"] == "generating_images":
            logger.info(f"[{job_id}] Generating Keyframes...")
            screenplay = state["screenplay"]

            with LocalImageProvider() as image_provider:
                for scene in screenplay["scenes"]:
                    sn = str(scene["scene_number"])
                    if sn in state.get("scene_images", {}):
                        logger.info(f"[{job_id}] Skipping image for scene {sn}, already generated.")
                        continue

                    img_prompt = scene["image_prompt"]
                    out_path = os.path.join(job_dir, f"scene_{sn}.png")
                    image_provider.generate(img_prompt, out_path)

                    if "scene_images" not in state:
                        state["scene_images"] = {}
                    state["scene_images"][sn] = out_path
                    self.checkpoint.save_state(job_id, state)

            cleanup_vram()

            state["status"] = "generating_videos"
            self.checkpoint.save_state(job_id, state)

        if state["status"] == "generating_videos":
            logger.info(f"[{job_id}] Generating Video Clips...")
            screenplay = state["screenplay"]

            with LocalVideoProvider() as video_provider:
                for scene in screenplay["scenes"]:
                    sn = str(scene["scene_number"])
                    if sn in state.get("scene_videos", {}):
                        logger.info(f"[{job_id}] Skipping video for scene {sn}, already generated.")
                        continue

                    vid_prompt = scene["video_prompt"]
                    out_path = os.path.join(job_dir, f"scene_{sn}.mp4")

                    video_provider.generate_from_text(vid_prompt, out_path)

                    if "scene_videos" not in state:
                        state["scene_videos"] = {}
                    state["scene_videos"][sn] = out_path
                    self.checkpoint.save_state(job_id, state)

            cleanup_vram()

            state["status"] = "assembling_video"
            self.checkpoint.save_state(job_id, state)

        if state["status"] == "assembling_video":
            logger.info(f"[{job_id}] Stitching clips with FFmpeg...")
            final_output = os.path.join(self.output_dir, f"{job_id}_final.mp4")

            scene_videos = [state["scene_videos"][str(s["scene_number"])] for s in state["screenplay"]["scenes"]]

            assembler = FFmpegAssemblyPipeline()
            assembler.assemble(scene_videos, final_output)

            state["final_video"] = final_output
            state["status"] = "completed"
            self.checkpoint.save_state(job_id, state)
            logger.info(f"[{job_id}] Pipeline complete. Output saved to {final_output}")

        return state["final_video"]

    def _assert_not_completed(self, state: Dict[str, Any], job_id: str):
        """Guard: if job is already completed with a valid video path, return early."""
        if state.get("status") == "completed":
            final = state.get("final_video")
            if final and os.path.exists(final):
                logger.info(f"[{job_id}] Already completed. Returning existing video: {final}")
                return final
            logger.warning(f"[{job_id}] State is 'completed' but video file missing. Re-running assembly stage.")
            state["status"] = "assembling_video"
        return None
