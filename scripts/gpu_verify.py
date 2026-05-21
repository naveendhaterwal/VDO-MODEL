import torch
import sys

def verify_gpu():
    print("--- GPU Verification ---")
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available.")
        sys.exit(1)
    
    device_count = torch.cuda.device_count()
    print(f"[OK] Found {device_count} GPUs.")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name}")
        print(f"  VRAM: {vram_gb:.2f} GB")
        
        if vram_gb < 22:
            print(f"[WARNING] GPU {i} has less than 22GB VRAM. OOM is highly likely.")
            
        capability = torch.cuda.get_device_capability(i)
        if capability[0] < 8:
            print(f"[WARNING] GPU {i} compute capability {capability[0]}.{capability[1]} < 8.0. Native bfloat16 not supported.")
            
    print("--- End Verification ---")

if __name__ == "__main__":
    verify_gpu()
