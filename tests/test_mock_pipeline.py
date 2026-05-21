import os
import shutil
import pytest
import fakeredis
from unittest import mock
from orchestration.pipeline import CinematicPipeline
from storage.checkpoint import CheckpointManager

def test_mock_inference_pipeline():
    # Force mock mode
    os.environ["MOCK_INFERENCE"] = "1"
    
    # Use a temporary output directory
    test_output_dir = "/tmp/test_outputs"
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
        
    with mock.patch("redis.Redis.from_url", lambda *a, **kw: fakeredis.FakeStrictRedis()):
        pipeline = CinematicPipeline(output_dir=test_output_dir)
        job_id = "test-mock-job"
        
        # Run the pipeline
        final_video_path = pipeline.run(job_id, "A cinematic shot of a testing pipeline")
    
    # Verify outputs
    assert os.path.exists(final_video_path)
    assert final_video_path.endswith(f"{job_id}_final.mp4")
    
    # Check that intermediate states were created
    job_dir = os.path.join(test_output_dir, job_id)
    assert os.path.exists(os.path.join(job_dir, "screenplay.json"))
    assert os.path.exists(os.path.join(job_dir, "scene_1.png"))
    assert os.path.exists(os.path.join(job_dir, "scene_1.mp4"))
    
    # Cleanup
    shutil.rmtree(test_output_dir)
