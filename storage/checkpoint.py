import json
import os
from redis import Redis
from typing import Dict, Any, Optional

class CheckpointManager:
    """
    Manages job state in Redis for resumable pipeline execution.
    """
    def __init__(self, redis_url: str = None):
        if not redis_url:
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self.redis = Redis.from_url(redis_url)
        
    def _key(self, job_id: str) -> str:
        return f"cinematic_job_state:{job_id}"
        
    def save_state(self, job_id: str, state: Dict[str, Any]):
        """Save the full state dictionary to Redis."""
        self.redis.set(self._key(job_id), json.dumps(state))
        
    def load_state(self, job_id: str) -> Dict[str, Any]:
        """Load state, or return an empty dict if none exists."""
        data = self.redis.get(self._key(job_id))
        if data:
            return json.loads(data)
        return {
            "status": "initialized",
            "prompt": "",
            "screenplay": None,
            "scene_images": {}, # scene_number -> image_path
            "scene_videos": {}, # scene_number -> video_path
            "final_video": None
        }
        
    def update_field(self, job_id: str, field: str, value: Any):
        """Update a specific top-level field in the state."""
        state = self.load_state(job_id)
        state[field] = value
        self.save_state(job_id, state)
        
    def clear_state(self, job_id: str):
        """Clear state (useful on complete success or hard reset)."""
        self.redis.delete(self._key(job_id))
