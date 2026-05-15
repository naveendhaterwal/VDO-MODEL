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
        state = checkpoint.load_state(job_id)
        state["last_heartbeat"] = time.time()
        checkpoint.save_state(job_id, state)
        stop_event.wait(30) # Update every 30 seconds

def generate_video_task(job_id: str, prompt: str):
    """
    RQ Task entrypoint.
    Runs the full cinematic pipeline for the given prompt.
    """
    logger.info(f"Task started: generate_video_task for job_id={job_id}")
    
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(target=heartbeat_loop, args=(job_id, stop_event))
    heartbeat_thread.start()
    
    try:
        pipeline = CinematicPipeline(output_dir="/app/outputs")
        final_video_path = pipeline.run(job_id, prompt)
        logger.info(f"Task completed successfully: {final_video_path}")
        return {"status": "completed", "video_path": final_video_path}
    except Exception as e:
        logger.error(f"Task failed: {e}")
        raise e
    finally:
        stop_event.set()
        heartbeat_thread.join()
        
        # Absolute guarantee that VRAM is released after every job (success or fail)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
