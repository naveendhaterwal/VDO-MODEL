import os
from huggingface_hub import snapshot_download, login

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Model list - all PUBLIC, no gated models required.
# FLUX.1-schnell is gated (requires HF approval). We use SDXL as the image backbone instead.
# Wan2.1-T2V-1.3B is the correct public repo ID for text-to-video.
MODELS_TO_DOWNLOAD = [
    "Qwen/Qwen2.5-7B-Instruct",                      # LLM for screenplay - PUBLIC
    "stabilityai/stable-diffusion-xl-base-1.0",       # Image gen - PUBLIC (replaces gated FLUX)
    "Wan-AI/Wan2.1-T2V-1.3B",                         # Video gen - PUBLIC
]

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Authenticate if HF_TOKEN is provided (needed for any gated models added later)
    if HF_TOKEN:
        print("Authenticating with HuggingFace...")
        login(token=HF_TOKEN)
    else:
        print("No HF_TOKEN set - downloading public models only.")

    print(f"Preloading models into {MODEL_DIR}...")

    failed = []
    for repo_id in MODELS_TO_DOWNLOAD:
        model_name = repo_id.split("/")[-1]
        local_dir = os.path.join(MODEL_DIR, model_name)
        print(f"Checking {repo_id}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=4,
                token=HF_TOKEN if HF_TOKEN else None
            )
            print(f"[OK] Successfully loaded {repo_id}")
        except Exception as e:
            print(f"[WARNING] Failed to download {repo_id}: {e}")
            failed.append(repo_id)

    if failed:
        print(f"\n[WARNING] The following models could not be downloaded: {failed}")
        print("The worker will attempt to load them at inference time.")
    else:
        print("\n[OK] All models preloaded successfully.")

if __name__ == "__main__":
    main()
