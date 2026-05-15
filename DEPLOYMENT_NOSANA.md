# Nosana Website Dashboard Deployment Guide

This guide provides the exact steps to deploy your local AI Cinematic Video platform to the Nosana decentralized GPU network using the **Nosana Web UI Dashboard**. 

> [!WARNING]
> This guide is optimized for a single-node RTX 4090 deployment. The architecture handles aggressive VRAM unloading to fit the 24GB limit. Do not configure jobs longer than 5-10 second clips initially to avoid worker timeouts.

## Phase 1: Local Docker Validation

Before deploying to Nosana, you must ensure the container builds and boots correctly on your local machine. 

1. **Build the Production Image:**
   (Note: If you are building on a Mac/Apple Silicon, you MUST specify the amd64 platform since Nosana GPU nodes are x86_64).
   ```bash
   cd /Users/n2/Desktop/nos/nos-vimax-wan
   docker build --platform linux/amd64 -f docker/Dockerfile.worker -t naveendhaterwal/vimax:latest .
   ```

2. **Test Locally:**
   ```bash
   docker run --gpus all -p 8000:8000 naveendhaterwal/vimax:latest
   ```
   *Watch the logs to ensure GPU validation passes, models download successfully, Redis boots, and Uvicorn starts the API on port 8000.*

3. **Push to a Public Registry:**
   Nosana nodes must be able to pull your image anonymously.
   ```bash
   docker push naveendhaterwal/vimax:latest
   ```

---

## Phase 2: Nosana Web Dashboard Deployment

Navigate to the [Nosana Job Dashboard](https://app.nosana.io) and connect your Solana wallet.

### 1. Select Compute Market
- Navigate to the **Markets** page.
- Select an **RTX 4090** market. (This architecture requires exactly 24GB of VRAM and CUDA support).

### 2. Create a New Job
Click **Create Job** and fill out the form using these exact parameters:

#### Container Configuration
- **Image URL**: `docker.io/naveendhaterwal/vimax:latest`
- **GPU Required**: `Enabled / Checked`
- **Cmd / Entrypoint**: Leave blank (it will use our robust `scripts/start_worker.sh` by default).

#### Environment Variables
You must set these for the container to function properly:
- `REDIS_URL` = `redis://localhost:6379` (Since the worker and Redis are packaged together in the single node).
- `MODEL_DIR` = `/app/models`

#### Network Expose
- **Exposed Port**: `8000`
- **Protocol**: `HTTP/TCP`

### 3. Review and Post
- Verify your wallet has enough SOL for gas and NOS for compute time.
- Click **Post Job**.

---

## Phase 3: Access & Monitoring

Once your job is picked up by a node, the Nosana dashboard will provide an active endpoint URL (e.g., `https://node-1234.nosana.network`).

### Expected Cold-Start Behavior
> [!IMPORTANT]
> The Nosana node starts from scratch. Upon boot, our `start_worker.sh` script will:
> 1. Validate the RTX 4090 and CUDA drivers.
> 2. Download ~20-30GB of weights from HuggingFace (Qwen2.5, Flux, Wan2.1).
> 
> **The API will not accept requests during this time.** Expect a 10-20 minute delay before the `/health` endpoint becomes responsive. Do not terminate the job prematurely.

### Connecting the Frontend
1. Open your Next.js project locally (`frontend/`).
2. Update the `fetch` URLs in `frontend/app/page.tsx` from `http://localhost:8000` to your new remote endpoint: `https://node-1234.nosana.network`.
3. Run `npm run build && npm start` to interact with your remote decentralized cinematic engine.

---

## Troubleshooting Guide

**1. Node fails immediately on boot**
- *Cause*: Likely a Docker pull failure or missing GPU on the node.
- *Fix*: Ensure your Docker Hub repository is public. Verify you selected an RTX 4090 market that enforces GPU flags.

**2. Out of Memory (OOM) Errors during Generation**
- *Cause*: A previous model failed to unload properly.
- *Fix*: The pipeline uses aggressive `gc.collect()` and `torch.cuda.empty_cache()`. If it still fails, the scene prompt may be too complex, or Wan 2.1 is generating too many frames. Keep clips to 5 seconds (approx 81 frames).

**3. API is running, but /generate returns 500**
- *Cause*: The internal Redis server or RQ worker thread died.
- *Fix*: Check the `/metrics` endpoint to view worker uptime. Ensure `REDIS_URL=redis://localhost:6379` was set in the Nosana dashboard.
