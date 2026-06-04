import os
import sys
import uuid
import logging
from unittest.mock import patch

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Trace")

# Ensure we are NOT in mock mode so real logic runs
os.environ.pop("MOCK_INFERENCE", None)
os.environ["OUTPUT_DIR"] = os.path.join(os.getcwd(), "trace_outputs")
os.makedirs(os.environ["OUTPUT_DIR"], exist_ok=True)

# Fake responses
FAKE_LLM_RESPONSE = """
Here is your cinematic screenplay:
```json
{
    "scenes": [
        {
            "scene_number": 1,
            "image_prompt": "A lone astronaut walking on the red sands of Mars, 8k, cinematic lighting",
            "video_prompt": "Slow pan across the Martian landscape, following the astronaut"
        },
        {
            "scene_number": 2,
            "image_prompt": "A huge dust storm approaching the astronaut, dramatic sky",
            "video_prompt": "The dust storm sweeps in rapidly, covering the screen"
        },
        {
            "scene_number": 3,
            "image_prompt": "Astronaut taking shelter behind a rock formation, close up",
            "video_prompt": "Camera zooms in on the astronaut's visor reflecting the storm"
        }
    ]
}
```
Enjoy your video!
"""

# Create mock classes
class MockChatProvider:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass
    def generate(self, prompt, system_prompt=None):
        logger.info("\n=== 🧠 CHAT PROVIDER (LLM) CALLED ===")
        logger.info(f"➡️  User Prompt: {prompt}")
        logger.info(f"➡️  System Prompt: {system_prompt}")
        logger.info(f"⬅️  Returning Mock Response:\n{FAKE_LLM_RESPONSE}")
        return FAKE_LLM_RESPONSE

class MockImageProvider:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass
    def generate(self, prompt, output_path):
        logger.info("\n=== 🖼️ IMAGE PROVIDER CALLED ===")
        logger.info(f"➡️  Image Prompt: {prompt}")
        logger.info(f"➡️  Output Path: {output_path}")
        logger.info(f"⬅️  Simulating saving image to {output_path}...")
        with open(output_path, "w") as f: f.write("mock image data")
        return output_path

class MockVideoProvider:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass
    def generate_from_text(self, prompt, output_path):
        logger.info("\n=== 🎥 VIDEO PROVIDER CALLED ===")
        logger.info(f"➡️  Video Prompt: {prompt}")
        logger.info(f"➡️  Output Path: {output_path}")
        logger.info(f"⬅️  Simulating saving video to {output_path}...")
        with open(output_path, "w") as f: f.write("mock video data")
        return output_path

class MockFFmpeg:
    def assemble(self, video_paths, output_path):
        logger.info("\n=== 🎞️ FFMPEG ASSEMBLER CALLED ===")
        logger.info(f"➡️  Input Videos: {video_paths}")
        logger.info(f"➡️  Output Path: {output_path}")
        logger.info(f"⬅️  Simulating final stitched video to {output_path}...")
        with open(output_path, "w") as f: f.write("mock final video data")
        return output_path

_MOCK_REDIS = {}
class MockCheckpointManager:
    def save_state(self, job_id, state):
        _MOCK_REDIS[job_id] = state
        logger.info(f"💾 State saved to Redis for {job_id}: status={state.get('status')}")
    def load_state(self, job_id):
        return _MOCK_REDIS.get(job_id, {})

def cleanup_vram_mock():
    pass

# Patch the imports BEFORE loading CinematicPipeline
from unittest.mock import MagicMock
sys.modules['providers.LocalChatProvider'] = MagicMock(LocalChatProvider=MockChatProvider)
sys.modules['providers.LocalImageProvider'] = MagicMock(LocalImageProvider=MockImageProvider)
sys.modules['providers.LocalVideoProvider'] = MagicMock(LocalVideoProvider=MockVideoProvider)

# Start Trace
logger.info("=========================================")
logger.info("🚀 STARTING REAL PIPELINE TRACE")
logger.info("=========================================")

with patch('storage.checkpoint.CheckpointManager', new=MockCheckpointManager), \
     patch('pipelines.ffmpeg_assembly.FFmpegAssemblyPipeline', new=MockFFmpeg), \
     patch('orchestration.pipeline.cleanup_vram', new=cleanup_vram_mock):

    from orchestration.pipeline import CinematicPipeline
    pipeline = CinematicPipeline(output_dir=os.environ["OUTPUT_DIR"])
    job_id = "trace-" + str(uuid.uuid4())[:8]
    prompt = "A cinematic sequence of an astronaut on Mars"
    
    try:
        final_video = pipeline.run(job_id, prompt)
        logger.info("\n✅ PIPELINE TRACE COMPLETED SUCCESSFULLY")
        logger.info(f"🎬 Final Video Path: {final_video}")
    except Exception:
        logger.exception("\n❌ PIPELINE CRASHED")
