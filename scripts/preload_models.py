import os
import time
from huggingface_hub import snapshot_download, login

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MODELS_TO_DOWNLOAD = [
    "Qwen/Qwen2.5-7B-Instruct",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
]


def download_with_retry(repo_id, local_dir, token, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                max_workers=8,
                ignore_patterns=[
                    "*.msgpack", "*.h5", "*.ot",
                    "*onnx*", "*openvino*", "*coreml*", "*flax*",
                ],
                token=token,
            )
            return True
        except Exception as e:
            print(f"[WARNING] Attempt {attempt + 1}/{max_retries} failed for {repo_id}: {e}")
            if attempt < max_retries - 1:
                print(f"[RETRY] Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"[ERROR] All {max_retries} attempts failed for {repo_id}")
                return False
    return False


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if HF_TOKEN:
        print("Authenticating with HuggingFace...")
        login(token=HF_TOKEN)

    print(f"Preloading models into {MODEL_DIR}...")

    failed = []
    for repo_id in MODELS_TO_DOWNLOAD:
        model_name = repo_id.split("/")[-1]
        local_dir = os.path.join(MODEL_DIR, model_name)

        # config.json is universal: exists for both diffusers pipelines and transformers models
        if os.path.exists(os.path.join(local_dir, "config.json")):
            print(f"[SKIP] {repo_id} already downloaded")
            continue

        print(f"Downloading {repo_id}...")
        success = download_with_retry(
            repo_id,
            local_dir,
            token=HF_TOKEN if HF_TOKEN else None
        )
        if success:
            print(f"[OK] Successfully loaded {repo_id}")
        else:
            print(f"[WARNING] Failed to download {repo_id}")
            failed.append(repo_id)

    if failed:
        print(f"\n[WARNING] The following models could not be downloaded: {failed}")
        print("The worker will attempt to load them at inference time.")
    else:
        print("\n[OK] All models preloaded successfully.")


if __name__ == "__main__":
    main()
