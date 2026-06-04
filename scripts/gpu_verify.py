import os
import subprocess
import sys
from importlib import metadata

import torch

from monitoring.gpu_runtime import format_runtime_banner, get_primary_gpu_runtime_info


def _get_package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _require_exact_version(package_name: str, env_name: str) -> None:
    expected_version = os.environ.get(env_name)
    if not expected_version:
        return

    actual_version = _get_package_version(package_name)
    if actual_version != expected_version:
        raise RuntimeError(
            f"{package_name} version mismatch: expected {expected_version}, got {actual_version}"
        )


def _print_nvidia_smi() -> None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print("[INFO] nvidia-smi:")
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")
    except Exception as exc:
        print(f"[WARNING] Unable to query nvidia-smi: {exc}")


def verify_gpu() -> None:
    print("========================================")
    print(" GPU Runtime Verification")
    print("========================================")
    print(f"[INFO] {format_runtime_banner()}")
    _print_nvidia_smi()

    info = get_primary_gpu_runtime_info()
    if not info.cuda_available:
        raise RuntimeError("CUDA is not available. Refusing to start.")

    expected_torch_cuda = os.environ.get("APP_EXPECTED_TORCH_CUDA")
    if expected_torch_cuda and info.torch_cuda_version != expected_torch_cuda:
        raise RuntimeError(
            f"PyTorch CUDA mismatch: expected {expected_torch_cuda}, got {info.torch_cuda_version}"
        )

    major = float(info.compute_capability.split('.')[0])
    if major >= 10 and os.environ.get("ALLOW_UNSUPPORTED_GPU") != "1":
        raise RuntimeError(
            f"Unsupported GPU architecture: Compute Capability {info.compute_capability}. "
            "PyTorch 2.6.0 only supports up to Hopper (sm_90). "
            "Please deploy on a node with an A100, H100, RTX 4090, L40S, or A6000."
        )

    min_vram_gb = float(os.environ.get("MIN_GPU_VRAM_GB", "40"))
    if info.total_vram_gb < min_vram_gb:
        raise RuntimeError(
            f"Insufficient VRAM: required {min_vram_gb:.0f}GB, found {info.total_vram_gb:.2f}GB"
        )

    _require_exact_version("transformers", "APP_EXPECTED_TRANSFORMERS")
    _require_exact_version("diffusers", "APP_EXPECTED_DIFFUSERS")
    _require_exact_version("bitsandbytes", "APP_EXPECTED_BITSANDBYTES")

    print(f"[OK] GPU name: {info.device_name}")
    print(f"[OK] PyTorch version: {info.torch_version}")
    print(f"[OK] CUDA version: {info.torch_cuda_version}")
    print(f"[OK] Compute capability: {info.compute_capability}")
    print(f"[OK] Total VRAM: {info.total_vram_gb:.2f} GB")
    print("========================================")


if __name__ == "__main__":
    try:
        verify_gpu()
    except Exception as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)
