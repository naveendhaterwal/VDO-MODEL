import logging
import threading
import time
import torch
import gc
from orchestration.pipeline import CinematicPipeline
from storage.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


def heartbeat_loop(job_id: str, stop_event: threading.Event):
    checkpoint = CheckpointManager()
    while not stop_event.is_set():
        try:
            state = checkpoint.load_state(job_id)
            state["last_heartbeat"] = time.time()
            checkpoint.save_state(job_id, state)
        except Exception as e:
            logger.warning(f"Heartbeat failed for {job_id}: {e}")
        stop_event.wait(30)


def generate_video_task(job_id: str, prompt: str):
    import os
    if os.environ.get("MOCK_INFERENCE") == "1":
        logger.warning("======================================================")
        logger.warning(f"Task started in MOCK_INFERENCE mode for job_id={job_id}")
        logger.warning("GPUs will NOT be used. Dummy videos will be generated.")
        logger.warning("======================================================")
    else:
        logger.info(f"Task started: generate_video_task for job_id={job_id}")

    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(target=heartbeat_loop, args=(job_id, stop_event), daemon=True)
    heartbeat_thread.start()

    try:
        pipeline = CinematicPipeline(output_dir=os.environ.get("OUTPUT_DIR", "/app/outputs"))
        final_video_path = pipeline.run(job_id, prompt)
        logger.info(f"Task completed successfully: {final_video_path}")
        return {"status": "completed", "video_path": final_video_path}
    except Exception as e:
        logger.error(f"Task failed: {e}")
        raise
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=10)

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
