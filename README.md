# CrewAI WAN2 Agent (Docker) — v1.0.0

**Copyright (c) RRECKTEK LLC**
**Built:** @EPOCH

A robust, containerized WAN2 LLM agent for generative video processing using Alibaba's Wan2.1/2.2 models, with agent-to-agent (A2A) messaging, multiple processing modes, and comprehensive Prometheus metrics monitoring.

## 🚀 Features

- **WAN2 Video Generation**: Text-to-Video using Wan2.1 (1.3B/14B) or Wan2.2 models
- **Diffusers Integration**: Built on HuggingFace diffusers library
- **Multiple Processing Modes**: Daemon, watch, and batch processing modes
- **Agent-to-Agent (A2A) API**: HTTP endpoints for job submission and inter-agent communication
- **Job Caching**: SHA256-based deduplication to avoid redundant processing
- **Prometheus Metrics**: 10+ comprehensive metrics for monitoring and alerting
- **pmem 1.0 Support**: Knowledge base (kb) for persistent memory
- **Robust Error Handling**: Comprehensive logging, fallbacks, and recovery
- **CUDA Acceleration**: GPU-optimized Docker image with CUDA 12.1

## 📋 Quick Start

```bash
# Build the container
cd crewai-wan2
docker build -t rrecktek/crewai-wan2:1.0.0 .

# Start daemon mode (recommended)
./run-wan2-agent-watch.sh daemon

# Check health and status
./run-wan2-agent-watch.sh status
./run-wan2-agent-watch.sh health
```

## 📁 Directory Structure

```
crewai-wan2/
├── input/                      # Place command JSON files here
│   ├── example_text_to_video.json
│   └── example_short_video.json
├── output/                     # Generated videos appear here
│   └── logs/                   # Processing logs and job cache
├── kb/                         # Knowledge base (pmem 1.0)
│   ├── short/                  # Atomic knowledge
│   └── long/                   # Emergent artifacts
├── app/
│   └── main.py                 # WAN2 agent script
├── metrics/
│   ├── wan2-dashboard.json     # Grafana dashboard
│   └── prometheus.yml          # Prometheus config
├── Dockerfile
├── run-wan2-agent-watch.sh     # Container management script
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `./input` | Directory for command JSON files |
| `OUTPUT_DIR` | `./output` | Output directory for videos |
| `WAN2_MODEL` | `Wan-AI/Wan2.1-T2V-1.3B` | WAN2 model to use |
| `API_PORT` | `8083` | A2A API server port |
| `METRICS_PORT` | `9093` | Prometheus metrics port |
| `JOB_TIMEOUT` | `3600` | Job timeout in seconds |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Container Ports

- **8083**: A2A API endpoints (health, job submission, status)
- **9093**: Prometheus metrics (`/metrics` endpoint)

## 🖥️ Usage

### Container Management

```bash
# Start daemon mode (background service)
./run-wan2-agent-watch.sh daemon

# Start watch mode (foreground, Ctrl-C to stop)
./run-wan2-agent-watch.sh watch

# One-shot batch processing
./run-wan2-agent-watch.sh batch

# Container lifecycle
./run-wan2-agent-watch.sh stop
./run-wan2-agent-watch.sh restart
./run-wan2-agent-watch.sh status
./run-wan2-agent-watch.sh health

# View logs
./run-wan2-agent-watch.sh logs
./run-wan2-agent-watch.sh logs -f  # Follow logs
```

### Submit Jobs

```bash
# Submit single command file
./run-wan2-agent-watch.sh job example_text_to_video.json

# Trigger batch processing of all JSON files
./run-wan2-agent-watch.sh trigger-batch
```

## 🔌 Agent-to-Agent (A2A) API

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/status` | Get agent status and metrics |
| `GET` | `/config` | Get current configuration |
| `POST` | `/job` | Submit video generation job |
| `POST` | `/batch` | Trigger batch processing |
| `POST` | `/config` | Update configuration |

### Examples

```bash
# Health check
curl http://localhost:8083/health

# Get status and metrics
curl http://localhost:8083/status | jq

# Submit text-to-video job (file-based)
curl -X POST http://localhost:8083/job \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/work/input/example_text_to_video.json", "force": false}'

# Submit text-to-video job (inline command)
curl -X POST http://localhost:8083/job \
  -H "Content-Type: application/json" \
  -d '{
    "command": {
      "type": "text_to_video",
      "prompt": "A dragon flying over mountains",
      "num_frames": 81,
      "height": 480,
      "width": 720,
      "num_inference_steps": 50,
      "guidance_scale": 7.5,
      "fps": 8,
      "output_format": "mp4"
    }
  }'

# Update WAN2 model configuration
curl -X POST http://localhost:8083/config \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Wan-AI/Wan2.2-T2V-A14B"}'

# Trigger batch processing
curl -X POST http://localhost:8083/batch \
  -H "Content-Type: application/json" \
  -d '{"force": false}'
```

## 🎬 Command File Format

Create JSON files in `input/` directory:

```json
{
  "type": "text_to_video",
  "prompt": "A cat walking on a beach at sunset, cinematic quality, 4k",
  "num_frames": 81,
  "height": 480,
  "width": 720,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "fps": 8,
  "output_format": "mp4"
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | `text_to_video` | Command type |
| `prompt` | string | required | Text description of video |
| `num_frames` | int | 81 | Number of frames to generate |
| `height` | int | 480 | Video height (480 or 720) |
| `width` | int | 720 | Video width |
| `num_inference_steps` | int | 50 | Generation steps (more=better quality) |
| `guidance_scale` | float | 7.5 | Prompt adherence (higher=more literal) |
| `fps` | int | 8 | Frames per second |
| `output_format` | string | `mp4` | Output format |

## 📊 Prometheus Metrics

Access metrics at `http://localhost:9093/metrics`

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `wan2_jobs_processed_total` | Counter | Successfully processed jobs |
| `wan2_jobs_failed_total` | Counter | Failed processing attempts |
| `wan2_jobs_skipped_total` | Counter | Jobs skipped (cached) |
| `wan2_processing_time_seconds_total` | Counter | Total processing time |
| `wan2_queue_depth` | Gauge | Jobs waiting in queue |
| `wan2_active_jobs` | Gauge | Currently processing jobs |
| `wan2_daemon_uptime_seconds` | Gauge | Agent uptime |
| `wan2_model_loads_total` | Counter | Model load count |
| `wan2_cache_hits_total` | Counter | Cache hit count |
| `wan2_last_processing_timestamp` | Gauge | Last job timestamp |

### Grafana Dashboard

Import `metrics/wan2-dashboard.json` into Grafana for visualization.

## 🎯 WAN2 Models

### Available Models (via Hugging Face)

| Model | Size | Resolution | Description |
|-------|------|------------|-------------|
| `Wan-AI/Wan2.1-T2V-1.3B` | 1.3B | 480P, 720P | Lightweight (8GB VRAM) |
| `Wan-AI/Wan2.1-T2V-14B` | 14B | 480P, 720P | High quality (80GB VRAM) |
| `Wan-AI/Wan2.2-T2V-A14B` | 14B | 480P, 720P | Latest with MoE |
| `Wan-AI/Wan2.2-TI2V-5B` | 5B | 720P | Text-Image-to-Video |

### Model Selection

Change model via environment variable:

```bash
export WAN2_MODEL="Wan-AI/Wan2.2-T2V-A14B"
./run-wan2-agent-watch.sh daemon
```

Or via API:

```bash
curl -X POST http://localhost:8083/config \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Wan-AI/Wan2.2-T2V-A14B"}'
```

## 💾 Hardware Requirements

| Model | VRAM | Recommended GPU |
|-------|------|-----------------|
| Wan2.1-T2V-1.3B | 8GB | RTX 3060, RTX 4060 |
| Wan2.1-T2V-14B | 80GB | A100, H100 |
| Wan2.2-T2V-A14B | 80GB | A100, H100 |

**CPU Mode**: Possible but very slow (hours per video)

## 🔍 Debugging

### Check Container Status

```bash
./run-wan2-agent-watch.sh status
./run-wan2-agent-watch.sh health
```

### View Logs

```bash
# Container logs
./run-wan2-agent-watch.sh logs -f

# Processing logs (inside container)
docker exec wan2-agent-daemon ls -la /work/output/logs/
docker exec wan2-agent-daemon tail -f /work/output/logs/*.log
```

### Common Issues

**CUDA/GPU not available:**
- Ensure nvidia-docker2 installed
- Check: `docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi`

**Out of memory:**
- Use smaller model (Wan2.1-T2V-1.3B)
- Reduce `num_frames`, `height`, or `width`
- Enable CPU offloading (automatic)

**Model download fails:**
- Check HuggingFace connectivity
- Models cache in `/work/.cache/huggingface/`

## 🚀 Production Deployment

### Docker Compose

```yaml
version: "3.9"
services:
  wan2-agent:
    image: rrecktek/crewai-wan2:1.0.0
    container_name: wan2-agent
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8083:8083"  # A2A API
      - "9093:9093"  # Prometheus metrics
    volumes:
      - ./input:/work/input
      - ./output:/work/output
      - ./kb:/work/kb
      - wan2-cache:/work/.cache/huggingface
    environment:
      - WAN2_MODEL=Wan-AI/Wan2.1-T2V-1.3B
      - LOG_LEVEL=INFO
    command: ["python3", "/opt/app/main.py", "--daemon"]

volumes:
  wan2-cache:
    driver: local
```

### Consul Registration

Register with crewai-c2 for service discovery:

```bash
# Set CONSUL_HTTP_ADDR to crewai-c2 host
export CONSUL_HTTP_ADDR=http://192.168.1.134:8500

# Agent auto-registers on startup
```

## 📝 License

Copyright (c) RRECKTEK LLC. All rights reserved.

## 📚 Sources

- [Wan2.1 GitHub](https://github.com/Wan-Video/Wan2.1)
- [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [Wan-AI on Hugging Face](https://huggingface.co/Wan-AI)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)

---

**For issues or questions, refer to the GitHub repositories above or RRECKTEK LLC support.**
