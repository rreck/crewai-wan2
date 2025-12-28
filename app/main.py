#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) RRECKTEK LLC
# Project: CrewAI WAN2 Agent
# Version: 1.0.0
# Built: @EPOCH
#
# Purpose:
#   WAN2 LLM agent for generative video processing and orchestration.
#   - Install and manage WAN2 LLM libraries
#   - Process primitives and commands from /input directory
#   - Execute generative video operations via WAN2 models
#   - Support both file-based and A2A API job submission
#   - SHA256-based job deduplication for caching
#   - Epoch-stamped outputs and per-job logs
#   - Daemon, watch, and batch processing modes
#   - Agent-to-Agent (A2A) HTTP API
#   - Prometheus metrics endpoint
#   - Consul service registration with crewai-c2
#
# Exit codes:
#   0 = success / no work
#   1 = general error
#   2 = at least one job failed
#   3 = daemon startup failed
# -----------------------------------------------------------------------------

import argparse
import atexit
import daemon
import daemon.pidfile
import hashlib
import json
import logging
import logging.handlers
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Tuple, Optional, Any
from urllib.parse import urlparse, parse_qs

# ---- Defaults (overridden by CLI or env) -------------------------------------

DEFAULT_INPUT = os.environ.get("INPUT_DIR", "./input")
DEFAULT_OUTPUT = os.environ.get("OUTPUT_DIR", "./output")
DEFAULT_PIDFILE = os.environ.get("PIDFILE", "/var/run/wan2-agent.pid")
DEFAULT_METRICS_PORT = int(os.environ.get("METRICS_PORT", "9093"))
DEFAULT_API_PORT = int(os.environ.get("API_PORT", "8083"))
DEFAULT_WAN2_MODEL = os.environ.get("WAN2_MODEL", "wan2-base")
JOB_TIMEOUT_SEC = int(os.environ.get("JOB_TIMEOUT", "3600"))

# ---- Global state for metrics and daemon control ----------------------------

class AgentMetrics:
    """Thread-safe metrics collection for Prometheus export"""
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics = {
            'jobs_processed_total': 0,
            'jobs_failed_total': 0,
            'jobs_skipped_total': 0,
            'processing_time_seconds_total': 0.0,
            'queue_depth': 0,
            'daemon_uptime_seconds': 0,
            'last_processing_timestamp': 0,
            'active_jobs': 0,
            'model_loads_total': 0,
            'cache_hits_total': 0
        }
        self._start_time = time.time()

    def increment(self, metric: str, value: float = 1.0):
        """Thread-safe metric increment"""
        with self._lock:
            if metric in self._metrics:
                self._metrics[metric] += value

    def set_gauge(self, metric: str, value: float):
        """Thread-safe gauge setting"""
        with self._lock:
            self._metrics[metric] = value

    def get_metrics(self) -> Dict[str, Any]:
        """Get snapshot of current metrics"""
        with self._lock:
            metrics = self._metrics.copy()
            metrics['daemon_uptime_seconds'] = time.time() - self._start_time
            return metrics

# Global metrics instance
METRICS = AgentMetrics()

# Daemon control
SHUTDOWN_EVENT = threading.Event()

# ---- Logging setup -----------------------------------------------------------

def setup_logging(daemon_mode: bool = False, log_level: str = "INFO"):
    """Configure logging for both file and syslog (in daemon mode)"""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Always log to file
    try:
        file_handler = logging.FileHandler('/var/log/wan2-agent.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (PermissionError, FileNotFoundError):
        # Fallback to local log file if /var/log not writable
        file_handler = logging.FileHandler('./wan2-agent.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if daemon_mode:
        # Add syslog handler for daemon mode
        try:
            syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
            syslog_formatter = logging.Formatter('wan2-agent: %(levelname)s - %(message)s')
            syslog_handler.setFormatter(syslog_formatter)
            logger.addHandler(syslog_handler)
        except Exception as e:
            logger.warning(f"Could not setup syslog handler: {e}")
    else:
        # Add console handler for non-daemon mode
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# ---- Directory and validation utilities --------------------------------------

def ensure_host_directories(*paths) -> bool:
    """Create directories on host with proper error checking"""
    logger = logging.getLogger(__name__)
    success = True

    for path in paths:
        try:
            os.makedirs(path, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(path, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"Directory verified: {path}")
        except PermissionError:
            logger.error(f"Permission denied creating/writing to directory: {path}")
            success = False
        except OSError as e:
            logger.error(f"OS error creating directory {path}: {e}")
            success = False
        except Exception as e:
            logger.error(f"Unexpected error with directory {path}: {e}")
            success = False

    return success

def validate_disk_space(path: str, min_mb: int = 1000) -> bool:
    """Check if path has minimum disk space available (1GB default for video)"""
    logger = logging.getLogger(__name__)
    try:
        stat = shutil.disk_usage(path)
        free_mb = stat.free // (1024 * 1024)
        if free_mb < min_mb:
            logger.warning(f"Low disk space in {path}: {free_mb}MB free (minimum {min_mb}MB)")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking disk space for {path}: {e}")
        return False

# ---- Small utils -------------------------------------------------------------

def epoch() -> int:
    return int(time.time())

def sha256_file(path: str) -> str:
    """Compute SHA256 hash of file"""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"
    except Exception as e:
        logging.getLogger(__name__).error(f"Error hashing file {path}: {e}")
        return f"sha256:error-{epoch()}"

def sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"

def safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in stem)

def run(cmd: List[str], timeout: int = JOB_TIMEOUT_SEC) -> Tuple[bool, str]:
    """Run a command; return (ok, combined_output)"""
    logger = logging.getLogger(__name__)
    try:
        logger.debug(f"Running command: {' '.join(cmd)}")
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return (p.returncode == 0, p.stdout)
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        logger.warning(f"Command timeout after {timeout}s: {' '.join(cmd)}")
        return (False, out + "\n[TIMEOUT]")
    except Exception as e:
        logger.error(f"Command execution error: {e}")
        return (False, f"[ERROR: {e}]")

def log_append(logf: str, text: str):
    """Append text to log file"""
    try:
        with open(logf, "a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    except Exception as e:
        logging.getLogger(__name__).error(f"Error writing to log file {logf}: {e}")

# ---- WAN2 Integration --------------------------------------------------------

class WAN2Manager:
    """Manage WAN2 model loading and video generation using diffusers"""

    def __init__(self, model_name: str = DEFAULT_WAN2_MODEL):
        self.model_name = model_name
        self.pipe = None
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """Load WAN2 model via diffusers"""
        if self.pipe is not None:
            return True

        self.logger.info(f"Loading WAN2 model: {self.model_name}")
        try:
            from diffusers import WanPipeline
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
            import torch

            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            self.logger.info(f"Using device: {device} with dtype: {dtype}")

            # Load pipeline (VAE is included automatically in Diffusers models)
            self.pipe = WanPipeline.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            )

            # Configure scheduler with flow_shift (5.0 for 720P, 3.0 for 480P)
            flow_shift = 3.0  # Default to 480P
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                flow_shift=flow_shift
            )

            # Move pipeline to target device
            self.pipe = self.pipe.to(device)

            METRICS.increment('model_loads_total')
            self.logger.info(f"WAN2 model {self.model_name} loaded successfully on {device}")
            return True
        except ImportError as e:
            self.logger.error(f"Failed to import diffusers: {e}")
            self.logger.error("Install with: pip install diffusers>=0.30.0")
            return False
        except Exception as e:
            self.logger.error(f"Failed to load WAN2 model: {e}")
            return False

    def process_command(self, command: Dict[str, Any], output_path: str) -> Tuple[bool, str]:
        """Process a WAN2 command and generate video"""
        if not self.load_model():
            return False, "Failed to load WAN2 model"

        try:
            cmd_type = command.get('type', 'text_to_video')
            self.logger.info(f"Processing WAN2 command: {cmd_type}")

            if cmd_type == 'text_to_video':
                # Text-to-Video generation
                prompt = command.get('prompt', '')
                num_frames = command.get('num_frames', 81)
                height = command.get('height', 480)
                width = command.get('width', 720)
                num_inference_steps = command.get('num_inference_steps', 50)
                guidance_scale = command.get('guidance_scale', 7.5)

                self.logger.info(f"Generating video from prompt: {prompt[:50]}...")

                # Generate video
                output = self.pipe(
                    prompt=prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )

                # Save video using diffusers export_to_video
                from diffusers.utils import export_to_video

                fps = command.get('fps', 8)
                video_frames = output.frames[0]
                export_to_video(video_frames, output_path, fps=fps)

                self.logger.info(f"Video saved to: {output_path}")
                return True, f"Generated {len(video_frames)} frames"

            else:
                return False, f"Unsupported command type: {cmd_type}"

        except Exception as e:
            self.logger.error(f"WAN2 processing error: {e}", exc_info=True)
            return False, str(e)

# Global WAN2 manager
wan2_manager = WAN2Manager()

# ---- Job processing ----------------------------------------------------------

def compute_job_key(input_path: str, model_name: str) -> str:
    """Compute unique job key for deduplication"""
    input_cid = sha256_file(input_path)
    combined = f"{input_cid}|{model_name}"
    return sha256_bytes(combined.encode("utf-8"))

def already_done(cache_dir: str, job_key: str) -> Optional[str]:
    """Return artifact path if job_key cached, else None"""
    idx = os.path.join(cache_dir, f"{job_key}.done")
    if os.path.isfile(idx):
        try:
            with open(idx, "r", encoding="utf-8") as f:
                p = f.read().strip()
            return p if p else None
        except Exception:
            return None
    return None

def mark_done(cache_dir: str, job_key: str, artifact_path: str):
    """Mark job as completed in cache"""
    try:
        with open(os.path.join(cache_dir, f"{job_key}.done"), "w", encoding="utf-8") as f:
            f.write(artifact_path)
    except Exception as e:
        logging.getLogger(__name__).error(f"Error writing cache file: {e}")

def process_one(
    input_path: str,
    output_dir: str,
    model_name: str,
    force: bool = False,
) -> Tuple[str, Optional[str], str]:
    """
    Process a WAN2 command file.
    Returns (status, artifact_path_or_None, log_path):
      status in {"OK","FAIL","SKIP"}
    """
    logger = logging.getLogger(__name__)
    METRICS.increment('active_jobs')

    try:
        stem = safe_stem(input_path)
        logs_dir = os.path.join(output_dir, "logs")
        cache_dir = os.path.join(logs_dir, "jobcache")

        if not ensure_host_directories(logs_dir, cache_dir):
            METRICS.increment('jobs_failed_total')
            return ("FAIL", None, "")

        logf = os.path.join(logs_dir, f"{epoch()}.{stem}.log")
        job_key = compute_job_key(input_path, model_name)

        logger.info(f"Processing {input_path} -> job_key={job_key}")

        # Dedupe/skip if cached and not forced
        if not force:
            prior = already_done(cache_dir, job_key)
            if prior and os.path.isfile(prior):
                log_append(logf, f"SKIP: cached job_key {job_key}; artifact={prior}")
                METRICS.increment('jobs_skipped_total')
                METRICS.increment('cache_hits_total')
                return ("SKIP", prior, logf)

        # Check disk space
        if not validate_disk_space(output_dir):
            log_append(logf, "FAIL: insufficient disk space")
            METRICS.increment('jobs_failed_total')
            return ("FAIL", None, logf)

        # Load command from input file
        try:
            with open(input_path, 'r') as f:
                command = json.load(f)
        except Exception as e:
            log_append(logf, f"FAIL: Could not parse command file: {e}")
            METRICS.increment('jobs_failed_total')
            return ("FAIL", None, logf)

        # Process with WAN2
        output_ext = command.get('output_format', 'mp4')
        output_path = os.path.join(output_dir, f"{epoch()}.{stem}.{output_ext}")

        start_time = time.time()
        ok, msg = wan2_manager.process_command(command, output_path)
        duration = time.time() - start_time

        METRICS.increment('processing_time_seconds_total', duration)

        if ok:
            mark_done(cache_dir, job_key, output_path)
            with open(output_path, "rb") as f:
                art_cid = sha256_bytes(f.read())
            log_append(logf, f"[ARTIFACT] output={output_path} cid={art_cid} duration={duration:.2f}s")
            logger.info(f"Successfully processed {input_path}")
            METRICS.increment('jobs_processed_total')
            METRICS.set_gauge('last_processing_timestamp', time.time())
            return ("OK", output_path, logf)
        else:
            log_append(logf, f"FATAL: WAN2 processing failed: {msg}")
            logger.error(f"Failed to process {input_path}: {msg}")
            METRICS.increment('jobs_failed_total')
            return ("FAIL", None, logf)

    finally:
        METRICS.increment('active_jobs', -1)

# ---- Batch / Watch / Daemon --------------------------------------------------

def list_command_files(input_dir: str) -> List[str]:
    """List all JSON command files in input directory"""
    try:
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".json")]
        files.sort()
        return files
    except FileNotFoundError:
        logging.getLogger(__name__).warning(f"Input directory not found: {input_dir}")
        return []
    except Exception as e:
        logging.getLogger(__name__).error(f"Error listing command files: {e}")
        return []

def batch(input_dir: str, output_dir: str, model_name: str, force: bool) -> int:
    """Process all command files in input directory once"""
    logger = logging.getLogger(__name__)
    files = list_command_files(input_dir)

    if not files:
        logger.info("No .json command files found in input/")
        return 0

    METRICS.set_gauge('queue_depth', len(files))
    summary = {"OK":0, "FAIL":0, "SKIP":0}

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=min(2, len(files))) as executor:
        futures = []
        for cmd_file in files:
            future = executor.submit(process_one, cmd_file, output_dir, model_name, force)
            futures.append((cmd_file, future))

        for cmd_file, future in futures:
            try:
                st, art, logf = future.result(timeout=JOB_TIMEOUT_SEC + 60)
                summary[st] += 1
                print(f"{st}\t{cmd_file}\t{art or ''}\t{logf}")
                logger.info(f"Batch result: {st} for {cmd_file}")
            except Exception as e:
                summary["FAIL"] += 1
                logger.error(f"Batch processing error for {cmd_file}: {e}")
                print(f"FAIL\t{cmd_file}\t\terror")

    METRICS.set_gauge('queue_depth', 0)
    logger.info(f"Batch complete: {summary}")
    return 0 if summary["FAIL"] == 0 else 2

def watch(input_dir: str, output_dir: str, model_name: str, sleep_sec: int, force: bool):
    """Watch mode - continuously monitor input directory"""
    logger = logging.getLogger(__name__)
    logger.info(f"[watch] scanning every {sleep_sec}s — Ctrl-C to stop")

    while not SHUTDOWN_EVENT.is_set():
        try:
            batch(input_dir, output_dir, model_name, force)

            # Sleep with early exit on shutdown
            for _ in range(max(1, sleep_sec)):
                if SHUTDOWN_EVENT.is_set():
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Watch mode interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in watch loop: {e}")
            time.sleep(5)

def daemon_main(input_dir: str, output_dir: str, model_name: str, sleep_sec: int, force: bool):
    """Main daemon loop"""
    logger = logging.getLogger(__name__)
    logger.info("Daemon mode started")

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        SHUTDOWN_EVENT.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        watch(input_dir, output_dir, model_name, sleep_sec, force)
    except Exception as e:
        logger.error(f"Daemon error: {e}")
        return 1

    logger.info("Daemon shutdown complete")
    return 0

# ---- Prometheus metrics server -----------------------------------------------

class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint"""

    def do_GET(self):
        """Handle GET requests for metrics"""
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.end_headers()

            metrics = METRICS.get_metrics()

            # Format metrics in Prometheus exposition format
            output = []
            output.append("# HELP wan2_jobs_processed_total Total number of jobs successfully processed")
            output.append("# TYPE wan2_jobs_processed_total counter")
            output.append(f"wan2_jobs_processed_total {metrics['jobs_processed_total']}")

            output.append("# HELP wan2_jobs_failed_total Total number of jobs that failed processing")
            output.append("# TYPE wan2_jobs_failed_total counter")
            output.append(f"wan2_jobs_failed_total {metrics['jobs_failed_total']}")

            output.append("# HELP wan2_jobs_skipped_total Total number of jobs skipped (cached)")
            output.append("# TYPE wan2_jobs_skipped_total counter")
            output.append(f"wan2_jobs_skipped_total {metrics['jobs_skipped_total']}")

            output.append("# HELP wan2_processing_time_seconds_total Total processing time in seconds")
            output.append("# TYPE wan2_processing_time_seconds_total counter")
            output.append(f"wan2_processing_time_seconds_total {metrics['processing_time_seconds_total']:.2f}")

            output.append("# HELP wan2_queue_depth Current number of jobs in processing queue")
            output.append("# TYPE wan2_queue_depth gauge")
            output.append(f"wan2_queue_depth {metrics['queue_depth']}")

            output.append("# HELP wan2_daemon_uptime_seconds Agent uptime in seconds")
            output.append("# TYPE wan2_daemon_uptime_seconds gauge")
            output.append(f"wan2_daemon_uptime_seconds {metrics['daemon_uptime_seconds']:.2f}")

            output.append("# HELP wan2_last_processing_timestamp Last job processing timestamp")
            output.append("# TYPE wan2_last_processing_timestamp gauge")
            output.append(f"wan2_last_processing_timestamp {metrics['last_processing_timestamp']}")

            output.append("# HELP wan2_active_jobs Current number of active processing jobs")
            output.append("# TYPE wan2_active_jobs gauge")
            output.append(f"wan2_active_jobs {metrics['active_jobs']}")

            output.append("# HELP wan2_model_loads_total Total number of model loads")
            output.append("# TYPE wan2_model_loads_total counter")
            output.append(f"wan2_model_loads_total {metrics['model_loads_total']}")

            output.append("# HELP wan2_cache_hits_total Total number of cache hits")
            output.append("# TYPE wan2_cache_hits_total counter")
            output.append(f"wan2_cache_hits_total {metrics['cache_hits_total']}")

            response = "\n".join(output) + "\n"
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        """Override to use proper logging"""
        logging.getLogger(__name__).info(f"Metrics request: {format % args}")

def start_metrics_server(port: int = DEFAULT_METRICS_PORT):
    """Start Prometheus metrics HTTP server in background thread"""
    logger = logging.getLogger(__name__)

    def run_server():
        try:
            server = HTTPServer(('0.0.0.0', port), MetricsHandler)
            logger.info(f"Metrics server started on port {port}")
            server.serve_forever()
        except Exception as e:
            logger.error(f"Metrics server error: {e}")

    metrics_thread = threading.Thread(target=run_server, daemon=True)
    metrics_thread.start()
    return metrics_thread

# ---- Agent-to-Agent (A2A) API server ----------------------------------------

class A2AHandler(BaseHTTPRequestHandler):
    """HTTP handler for Agent-to-Agent communication API"""

    def do_GET(self):
        """Handle GET requests for status and queries"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == "/status":
            self.send_json_response({
                "status": "running",
                "metrics": METRICS.get_metrics(),
                "timestamp": time.time()
            })
        elif path == "/health":
            self.send_json_response({"status": "healthy"})
        elif path == "/config":
            config = {
                "input_dir": getattr(self.server, 'input_dir', 'unknown'),
                "output_dir": getattr(self.server, 'output_dir', 'unknown'),
                "model_name": getattr(self.server, 'model_name', DEFAULT_WAN2_MODEL),
            }
            self.send_json_response(config)
        else:
            self.send_error_response(404, "Endpoint not found")

    def do_POST(self):
        """Handle POST requests for job submission and configuration"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
        except Exception as e:
            self.send_error_response(400, f"Invalid JSON: {e}")
            return

        if path == "/job":
            result = self.handle_job_request(data)
            self.send_json_response(result)
        elif path == "/batch":
            result = self.handle_batch_request(data)
            self.send_json_response(result)
        elif path == "/config":
            result = self.handle_config_update(data)
            self.send_json_response(result)
        else:
            self.send_error_response(404, "Endpoint not found")

    def handle_job_request(self, data: dict) -> dict:
        """Handle individual job processing request"""
        logger = logging.getLogger(__name__)

        # Support both file_path (existing file) and inline command
        if 'file_path' in data:
            file_path = data['file_path']
            force = data.get('force', False)

            if not os.path.isfile(file_path) or not file_path.lower().endswith('.json'):
                return {"error": "Invalid JSON command file", "file_path": file_path}

            try:
                output_dir = getattr(self.server, 'output_dir', DEFAULT_OUTPUT)
                model_name = getattr(self.server, 'model_name', DEFAULT_WAN2_MODEL)

                status, artifact, log_path = process_one(file_path, output_dir, model_name, force)

                logger.info(f"A2A job request processed: {file_path} -> {status}")

                return {
                    "status": status,
                    "file_path": file_path,
                    "artifact_path": artifact,
                    "log_path": log_path,
                    "timestamp": time.time()
                }
            except Exception as e:
                logger.error(f"A2A job processing error: {e}")
                return {"error": str(e), "file_path": file_path}

        elif 'command' in data:
            # Inline command submission
            try:
                output_dir = getattr(self.server, 'output_dir', DEFAULT_OUTPUT)
                model_name = data.get('model_name', getattr(self.server, 'model_name', DEFAULT_WAN2_MODEL))

                # Create temporary command file
                temp_file = os.path.join(output_dir, f"temp_cmd_{epoch()}.json")
                with open(temp_file, 'w') as f:
                    json.dump(data['command'], f)

                status, artifact, log_path = process_one(temp_file, output_dir, model_name, data.get('force', False))

                # Remove temp file
                try:
                    os.remove(temp_file)
                except:
                    pass

                logger.info(f"A2A inline command processed -> {status}")

                return {
                    "status": status,
                    "artifact_path": artifact,
                    "log_path": log_path,
                    "timestamp": time.time()
                }
            except Exception as e:
                logger.error(f"A2A inline command error: {e}")
                return {"error": str(e)}
        else:
            return {"error": "Missing required field: 'file_path' or 'command'"}

    def handle_batch_request(self, data: dict) -> dict:
        """Handle batch processing request"""
        logger = logging.getLogger(__name__)

        try:
            input_dir = data.get('input_dir', getattr(self.server, 'input_dir', DEFAULT_INPUT))
            output_dir = getattr(self.server, 'output_dir', DEFAULT_OUTPUT)
            model_name = getattr(self.server, 'model_name', DEFAULT_WAN2_MODEL)
            force = data.get('force', False)

            # Run batch in background thread
            def run_batch():
                return batch(input_dir, output_dir, model_name, force)

            batch_thread = threading.Thread(target=run_batch)
            batch_thread.start()

            logger.info(f"A2A batch request triggered for {input_dir}")

            return {
                "status": "started",
                "message": "Batch processing started in background",
                "input_dir": input_dir,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"A2A batch request error: {e}")
            return {"error": str(e)}

    def handle_config_update(self, data: dict) -> dict:
        """Handle configuration update request"""
        logger = logging.getLogger(__name__)

        try:
            updated_fields = []

            if 'model_name' in data:
                self.server.model_name = data['model_name']
                wan2_manager.model_name = data['model_name']
                wan2_manager.model = None  # Force reload
                updated_fields.append('model_name')

            logger.info(f"A2A config updated: {updated_fields}")

            return {
                "status": "updated",
                "updated_fields": updated_fields,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"A2A config update error: {e}")
            return {"error": str(e)}

    def send_json_response(self, data: dict):
        """Send JSON response"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode('utf-8'))

    def send_error_response(self, code: int, message: str):
        """Send error response"""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        error_data = {"error": message, "timestamp": time.time()}
        response = json.dumps(error_data)
        self.wfile.write(response.encode('utf-8'))

    def log_message(self, format, *args):
        """Override to use proper logging"""
        logging.getLogger(__name__).info(f"A2A API request: {format % args}")

def start_a2a_server(port: int = DEFAULT_API_PORT, **server_config):
    """Start A2A API HTTP server in background thread"""
    logger = logging.getLogger(__name__)

    def run_server():
        try:
            server = HTTPServer(('0.0.0.0', port), A2AHandler)
            for key, value in server_config.items():
                setattr(server, key, value)

            logger.info(f"A2A API server started on port {port}")
            server.serve_forever()
        except Exception as e:
            logger.error(f"A2A API server error: {e}")

    api_thread = threading.Thread(target=run_server, daemon=True)
    api_thread.start()
    return api_thread

# ---- PID file management -----------------------------------------------------

@contextmanager
def pid_file_manager(pidfile_path: str):
    """Context manager for PID file creation and cleanup"""
    logger = logging.getLogger(__name__)

    try:
        if os.path.exists(pidfile_path):
            try:
                with open(pidfile_path, 'r') as f:
                    old_pid = int(f.read().strip())
                try:
                    os.kill(old_pid, 0)
                    logger.error(f"Daemon already running with PID {old_pid}")
                    raise SystemExit(3)
                except ProcessLookupError:
                    os.remove(pidfile_path)
                    logger.info(f"Removed stale PID file: {pidfile_path}")
            except (ValueError, FileNotFoundError):
                pass

        pid = os.getpid()
        with open(pidfile_path, 'w') as f:
            f.write(str(pid))
        logger.info(f"Created PID file: {pidfile_path} (PID: {pid})")

        def cleanup_pid():
            try:
                if os.path.exists(pidfile_path):
                    os.remove(pidfile_path)
                    logger.info(f"Removed PID file: {pidfile_path}")
            except Exception as e:
                logger.error(f"Error removing PID file: {e}")

        atexit.register(cleanup_pid)
        yield
    except Exception as e:
        logger.error(f"PID file management error: {e}")
        raise

# ---- CLI and main entry point ------------------------------------------------

def main() -> int:
    """Main entry point"""
    ap = argparse.ArgumentParser(
        description="CrewAI WAN2 Agent with daemon mode, A2A messaging, and Prometheus metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One-shot batch processing
  python main.py -i ./input -o ./output

  # Foreground watch mode
  python main.py -w -s 5

  # Daemon mode with custom ports
  python main.py --daemon --metrics-port 9093 --api-port 8083

  # Force re-processing
  python main.py -f

A2A API Endpoints:
  GET  /status      - Get agent status and metrics
  GET  /health      - Health check
  GET  /config      - Get current configuration
  POST /job         - Process specific file or inline command
  POST /batch       - Trigger batch processing
  POST /config      - Update configuration

Metrics available at http://localhost:9093/metrics (Prometheus format)
        """
    )

    ap.add_argument("-i","--input", default=DEFAULT_INPUT, help="input directory (JSON command files)")
    ap.add_argument("-o","--output", default=DEFAULT_OUTPUT, help="output directory")
    ap.add_argument("-m","--model", default=DEFAULT_WAN2_MODEL, help="WAN2 model name")
    ap.add_argument("-w","--watch", action="store_true", help="foreground watch mode")
    ap.add_argument("--daemon", action="store_true", help="daemon mode (background service)")
    ap.add_argument("-s","--sleep", type=int, default=5, help="seconds between scans in watch/daemon mode")
    ap.add_argument("-f","--force", action="store_true", help="force re-processing (ignore cache)")
    ap.add_argument("--pidfile", default=DEFAULT_PIDFILE, help="PID file path for daemon mode")
    ap.add_argument("--metrics-port", type=int, default=DEFAULT_METRICS_PORT, help="Prometheus metrics server port")
    ap.add_argument("--api-port", type=int, default=DEFAULT_API_PORT, help="A2A API server port")
    ap.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help="logging level")

    args = ap.parse_args()

    if args.watch and args.daemon:
        print("ERROR: Cannot use both --watch and --daemon modes simultaneously", file=sys.stderr)
        return 1

    setup_logging(daemon_mode=args.daemon, log_level=args.log_level)
    logger = logging.getLogger(__name__)

    if not ensure_host_directories(args.input, args.output, os.path.join(args.output, "logs")):
        logger.error("Failed to create or access required directories")
        return 2

    logger.info(f"Starting WAN2 agent with model: {args.model}")
    logger.info(f"Starting metrics server on port {args.metrics_port}")
    metrics_thread = start_metrics_server(args.metrics_port)

    logger.info(f"Starting A2A API server on port {args.api_port}")
    api_thread = start_a2a_server(
        args.api_port,
        input_dir=args.input,
        output_dir=args.output,
        model_name=args.model
    )

    try:
        if args.daemon:
            logger.info("Starting daemon mode")
            with pid_file_manager(args.pidfile):
                with daemon.DaemonContext(
                    pidfile=daemon.pidfile.TimeoutPIDLockFile(args.pidfile),
                    detach_process=True,
                ):
                    setup_logging(daemon_mode=True, log_level=args.log_level)
                    return daemon_main(args.input, args.output, args.model, args.sleep, args.force)

        elif args.watch:
            logger.info("Starting watch mode")
            watch(args.input, args.output, args.model, args.sleep, args.force)
            return 0

        else:
            logger.info("Starting batch mode")
            return batch(args.input, args.output, args.model, args.force)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
