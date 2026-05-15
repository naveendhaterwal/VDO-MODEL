import torch
import gc
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class MemoryWatchdog:
    """
    Monitors and manages CUDA memory to prevent OOM errors on the RTX 4090.
    """
    
    @staticmethod
    def get_vram_usage() -> Tuple[float, float]:
        """Returns (allocated_gb, reserved_gb)"""
        if not torch.cuda.is_available():
            return (0.0, 0.0)
            
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return allocated, reserved
        
    @staticmethod
    def enforce_cleanup():
        """
        Aggressive garbage collection and VRAM cache clearing.
        Should be called between pipeline stages.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        alloc, res = MemoryWatchdog.get_vram_usage()
        logger.info(f"[Watchdog] Memory cleaned. Allocated: {alloc:.2f}GB, Reserved: {res:.2f}GB")
        
    @staticmethod
    def assert_vram_available(required_gb: float = 18.0):
        """
        Ensures there is enough VRAM before starting a heavy model load.
        RTX 4090 has ~24GB. If reserved memory is high, it might fail.
        """
        if not torch.cuda.is_available():
            return
            
        alloc, res = MemoryWatchdog.get_vram_usage()
        # Assume 24GB total. Available is ~24 - reserved
        # Actually torch.cuda.mem_get_info() returns (free, total)
        free, total = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
        
        if free_gb < required_gb:
            logger.warning(f"Low VRAM! Only {free_gb:.2f}GB free. Running emergency cleanup.")
            MemoryWatchdog.enforce_cleanup()
            
            free_after, _ = torch.cuda.mem_get_info()
            free_gb_after = free_after / (1024**3)
            if free_gb_after < required_gb:
                raise RuntimeError(f"OOM Risk: Cannot free enough VRAM. Required: {required_gb}GB, Free: {free_gb_after:.2f}GB")
