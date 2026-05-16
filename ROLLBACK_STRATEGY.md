# Rollback Strategy

## Deployment Rollback

### Container Image Rollback

```bash
# Rollback to previous version
docker pull naveendhaterwal/vimax:v<previous-run-number>
docker tag naveendhaterwal/vimax:v<previous-run-number> naveendhaterwal/vimax:latest
docker push naveendhaterwal/vimax:latest

# On Nosana, update the image tag in the job configuration
```

### CI/CD Rollback

1. Revert the GitHub commit that introduced the breaking change
2. The `docker-publish.yml` workflow will rebuild from the previous commit
3. A new image with the fixed code will be pushed
4. Update Nosana job to reference the new image

### Quick Rollback via Tag Pinning

Always deploy with an explicit version tag, not `latest`:
```json
"image": "docker.io/naveendhaterwal/vimax:v42"
```

Keep `latest` pointing to the last known-good version:
```bash
# Mark current good version
docker tag naveendhaterwal/vimax:v42 naveendhaterwal/vimax:stable
docker push naveendhaterwal/vimax:stable

# After bad deploy to latest, rollback:
docker tag naveendhaterwal/vimax:stable naveendhaterwal/vimax:latest
docker push naveendhaterwal/vimax:latest
```

## Data Rollback

### Job State Recovery

If Redis data is lost (container restart), jobs can be recovered from checkpoints:

1. **Before restart:** Redis persistence should be enabled (appendonly yes)
2. **After restart:** The `CheckpointManager` returns `initialized` state for unknown jobs
3. **In-flight jobs are lost** — the client must resubmit

### Output Recovery

Output videos are stored at `/app/outputs/<job_id>_final.mp4`. If using persistent volumes:
- Files survive container restarts
- Old files should be cleaned via a TTL mechanism (cron job deleting files older than 7 days)

## Client-Side Error Handling

The frontend already handles:
- Job submission failure (shows "Failed to connect to compute node")
- Job status polling (every 5s)
- Job failure (shows "Pipeline execution failed on node")
- Video download (once ready)

For rollback, clients should:
1. Retry on 503 responses (healthcheck failure)
2. Use exponential backoff for status polling (5s → 10s → 20s → max 60s)
3. Handle 404 on `/result` gracefully (video not yet available)

## Nosana-Specific Rollback

1. **For single-job deployments:** Stop the current job and start a new one with the previous image tag
2. **For persistent service deployments:** Update the image tag in the deployment-manager config and trigger a redeploy
3. **Market failure:** If no nodes bid on the job, the market may need to be changed (e.g., from RTX 4090 to A100)
