import os
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Optional

import torch


@dataclass(frozen=True)
class GpuRuntimeInfo:
    torch_version: str
    torch_cuda_version: str
    cuda_available: bool
    device_count: int
    device_name: str
    compute_capability: str
    total_vram_gb: float
    supports_bf16: bool
    supports_4bit: bool


def _safe_package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def get_primary_gpu_runtime_info() -> GpuRuntimeInfo:
    torch_cuda_version = torch.version.cuda or "unknown"
    if not torch.cuda.is_available():
        return GpuRuntimeInfo(
            torch_version=torch.__version__,
            torch_cuda_version=torch_cuda_version,
            cuda_available=False,
            device_count=0,
            device_name="unavailable",
            compute_capability="unavailable",
            total_vram_gb=0.0,
            supports_bf16=False,
            supports_4bit=False,
        )

    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_vram_gb = props.total_memory / (1024**3)
    supports_bf16 = major >= 8 and torch.cuda.is_bf16_supported()
    supports_4bit = major >= 8 and os.environ.get("ENABLE_BNB_4BIT", "1") == "1"

    return GpuRuntimeInfo(
        torch_version=torch.__version__,
        torch_cuda_version=torch_cuda_version,
        cuda_available=True,
        device_count=torch.cuda.device_count(),
        device_name=props.name,
        compute_capability=f"{major}.{minor}",
        total_vram_gb=total_vram_gb,
        supports_bf16=supports_bf16,
        supports_4bit=supports_4bit,
    )


def get_preferred_torch_dtype() -> torch.dtype:
    info = get_primary_gpu_runtime_info()
    return torch.bfloat16 if info.supports_bf16 else torch.float16


def should_enable_4bit_quantization() -> bool:
    return get_primary_gpu_runtime_info().supports_4bit


def format_runtime_banner() -> str:
    info = get_primary_gpu_runtime_info()
    bitsandbytes_version = _safe_package_version("bitsandbytes")
    transformers_version = _safe_package_version("transformers")
    diffusers_version = _safe_package_version("diffusers")
    return (
        f"PyTorch={info.torch_version} | "
        f"Torch CUDA={info.torch_cuda_version} | "
        f"GPU={info.device_name} | "
        f"CC={info.compute_capability} | "
        f"VRAM={info.total_vram_gb:.2f}GB | "
        f"BF16={info.supports_bf16} | "
        f"4bit={info.supports_4bit} | "
        f"Transformers={transformers_version} | "
        f"Diffusers={diffusers_version} | "
        f"bitsandbytes={bitsandbytes_version} | "
        f"Python={sys.version.split()[0]}"
    )
