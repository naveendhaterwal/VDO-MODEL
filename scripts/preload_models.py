import json
import os
import shutil
import sys
import time
from glob import glob
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
MAX_RETRIES = int(os.environ.get("MODEL_DOWNLOAD_RETRIES", "3"))
RETRY_DELAY_SECONDS = int(os.environ.get("MODEL_DOWNLOAD_RETRY_DELAY_SECONDS", "5"))
PARTIAL_MARKER = ".partial-download"


def _has_any(path: Path, patterns: list[str]) -> bool:
    return any(glob(str(path / pattern), recursive=True) for pattern in patterns)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_transformers_model(local_dir: Path) -> None:
    AutoConfig.from_pretrained(local_dir, local_files_only=True)
    AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    _require((local_dir / "config.json").is_file(), f"Missing config.json in {local_dir}")
    _require(
        _has_any(local_dir, ["*.safetensors", "*.bin", "*.safetensors.index.json"]),
        f"Missing transformer weight files in {local_dir}",
    )


def validate_diffusers_model(local_dir: Path) -> None:
    model_index_path = local_dir / "model_index.json"
    _require(model_index_path.is_file(), f"Missing model_index.json in {local_dir}")

    with model_index_path.open() as handle:
        model_index = json.load(handle)

    component_names = [
        key for key, value in model_index.items()
        if not key.startswith("_") and isinstance(value, list)
    ]
    _require(component_names, f"No diffusers components declared in {model_index_path}")

    weight_components = {"unet", "vae", "transformer", "text_encoder", "text_encoder_2"}

    for component in component_names:
        component_dir = local_dir / component
        _require(component_dir.exists(), f"Missing component directory {component_dir}")
        _require(
            _has_any(component_dir, ["*.json", "*.txt", "*.model"]),
            f"Missing configuration files for component {component_dir}",
        )
        if component in weight_components:
            _require(
                _has_any(component_dir, ["*.safetensors", "*.bin", "*.index.json"]),
                f"Missing weight files for component {component_dir}",
            )


MODELS: list[dict[str, object]] = [
    {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "local_name": "Qwen2.5-7B-Instruct",
        "validator": validate_transformers_model,
    },
    {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "local_name": "stable-diffusion-xl-base-1.0",
        "validator": validate_diffusers_model,
    },
    {
        "repo_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "local_name": "Wan2.1-T2V-1.3B-Diffusers",
        "validator": validate_diffusers_model,
    },
]


def _partial_marker(local_dir: Path) -> Path:
    return local_dir / PARTIAL_MARKER


def _clear_partial_download(local_dir: Path) -> None:
    if local_dir.exists():
        print(f"[CLEANUP] Removing partial or invalid model directory: {local_dir}")
        shutil.rmtree(local_dir)


def _prepare_clean_directory(local_dir: Path, validator: Callable[[Path], None]) -> bool:
    marker = _partial_marker(local_dir)
    if marker.exists():
        _clear_partial_download(local_dir)
        return False

    if not local_dir.exists():
        return False

    try:
        validator(local_dir)
        print(f"[OK] Verified existing model cache: {local_dir}")
        return True
    except Exception as exc:
        print(f"[WARNING] Existing model cache is incomplete: {local_dir} ({exc})")
        _clear_partial_download(local_dir)
        return False


def _download_model(repo_id: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    marker = _partial_marker(local_dir)
    marker.write_text("download-in-progress\n")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=HF_TOKEN,
            max_workers=4,
            local_dir_use_symlinks=False,
            resume_download=False,
            ignore_patterns=[
                "*.msgpack",
                "*.h5",
                "*.ot",
                "*onnx*",
                "*openvino*",
                "*coreml*",
                "*flax*",
            ],
        )
    except Exception:
        _clear_partial_download(local_dir)
        raise
    else:
        marker.unlink(missing_ok=True)


def ensure_model(repo_id: str, local_name: str, validator: Callable[[Path], None]) -> None:
    local_dir = MODEL_DIR / local_name
    if _prepare_clean_directory(local_dir, validator):
        return

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[DOWNLOAD] {repo_id} -> {local_dir} (attempt {attempt}/{MAX_RETRIES})")
            _download_model(repo_id, local_dir)
            validator(local_dir)
            print(f"[OK] Model ready: {repo_id}")
            return
        except Exception as exc:
            print(f"[WARNING] Download attempt failed for {repo_id}: {exc}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Model preload failed for {repo_id}") from exc
            time.sleep(RETRY_DELAY_SECONDS)


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    for model in MODELS:
        try:
            ensure_model(
                repo_id=str(model["repo_id"]),
                local_name=str(model["local_name"]),
                validator=model["validator"],
            )
        except Exception as exc:
            failures.append(f"{model['repo_id']}: {exc}")

    if failures:
        print("[FAIL] Model preload failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("[OK] All required models are present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
