# Runtime Monitoring Checklist

## Container Health

- [ ] HEALTHCHECK at `/health` returns `{"status":"ok","redis":"connected"}`
- [ ] Healthcheck passes within 120s (start-period) after cold start
- [ ] Prometheus `/metrics` endpoint returns valid metrics
- [ ] RQ worker is alive (check via rq-dashboard or Redis keys)

## Metrics to Watch

| Metric | Type | Alert Threshold | Action |
|--------|------|----------------|--------|
| `api_requests_total` | Counter | N/A | Track usage trends |
| `api_request_latency_seconds` | Histogram | p99 > 30s | Check worker saturation |
| `jobs_created_total` | Counter | N/A | Track job volume |
| `torch.cuda.memory_allocated()` | Gauge | > 22GB | OOM risk, reduce quality |
| Job failure rate | Derived | > 10% | Investigate pipeline issues |

## Log Monitoring

- Monitor for: `OOM Risk`, `Task failed`, `GPU verification failed`
- Monitor for: `Failed to download`, `Segfault` (hf_transfer issue)
- Monitor for: `Could not enable xformers` (performance degradation)

## Redis Monitoring

- [ ] Check `INFO` for connected clients (should be 2: API + RQ)
- [ ] Check memory usage (`used_memory_human`)
- [ ] Verify heartbeat keys in Redis: `cinematic_job_state:*`
- [ ] Monitor for expired/scheduled jobs in RQ

## GPU Monitoring

- [ ] nvidia-smi periodically: VRAM usage, temperature, power draw
- [ ] Watch for ECC errors
- [ ] Watch for GPU compute mode changes

## Error Alerting

- [ ] Job failure → alert (check `rq:job:*` status)
- [ ] Redis disconnect → critical alert
- [ ] GPU verify failure → critical alert
- [ ] Healthcheck fails 3x → container restart (Kubernetes/Nosana)
- [ ] Disk space < 10GB → warning (output files accumulate)
