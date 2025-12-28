#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# CrewAI WAN2 Agent Container Management Script
# Copyright (c) RRECKTEK LLC
# Version: 1.0.0
#
# This script provides comprehensive management for the CrewAI WAN2 Agent
# container including daemon mode, health checks, and API integration.
# ------------------------------------------------------------------------------

set -euo pipefail

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly IMAGE_NAME="${IMAGE_NAME:-rrecktek/crewai-wan2:1.0.0}"
readonly CONTAINER_NAME="${CONTAINER_NAME:-wan2-agent}"
readonly DAEMON_CONTAINER_NAME="${DAEMON_CONTAINER_NAME:-wan2-agent-daemon}"

# Ports
readonly API_PORT="${API_PORT:-8083}"
readonly METRICS_PORT="${METRICS_PORT:-9093}"

# Directories
readonly INPUT_DIR="${INPUT_DIR:-${SCRIPT_DIR}/input}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
readonly APP_DIR="${APP_DIR:-${SCRIPT_DIR}/app}"
readonly KB_DIR="${KB_DIR:-${SCRIPT_DIR}/kb}"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Check if container exists
container_exists() {
    docker ps -a --format "table {{.Names}}" | grep -q "^${1}$" 2>/dev/null
}

# Check if container is running
container_running() {
    docker ps --format "table {{.Names}}" | grep -q "^${1}$" 2>/dev/null
}

# Wait for container to be healthy
wait_for_health() {
    local container_name="$1"
    local max_attempts=30
    local attempt=1

    log_info "Waiting for container $container_name to be healthy..."

    while [ $attempt -le $max_attempts ]; do
        if container_running "$container_name"; then
            if curl -s "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
                log_success "Container $container_name is healthy!"
                return 0
            fi
        fi

        echo -n "."
        sleep 2
        ((attempt++))
    done

    log_error "Container $container_name failed to become healthy within $((max_attempts * 2)) seconds"
    return 1
}

# Ensure directories exist
setup_directories() {
    mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "${OUTPUT_DIR}/logs" "${KB_DIR}/short" "${KB_DIR}/long"
    log_success "Directories verified"
}

# Build image if needed
ensure_image() {
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        log_warning "Image $IMAGE_NAME not found, building..."
        docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
        log_success "Image $IMAGE_NAME built successfully"
    fi
}

# Start daemon mode
start_daemon() {
    setup_directories
    ensure_image

    if container_running "$DAEMON_CONTAINER_NAME"; then
        log_warning "Daemon container $DAEMON_CONTAINER_NAME is already running"
        return 0
    fi

    if container_exists "$DAEMON_CONTAINER_NAME"; then
        log_info "Removing existing daemon container..."
        docker rm "$DAEMON_CONTAINER_NAME" >/dev/null 2>&1
    fi

    log_info "Starting WAN2 Agent in daemon mode..."
    docker run -d \
        --name "$DAEMON_CONTAINER_NAME" \
        --restart unless-stopped \
        -v "${INPUT_DIR}:/work/input" \
        -v "${OUTPUT_DIR}:/work/output" \
        -v "${APP_DIR}:/opt/app" \
        -v "${KB_DIR}:/work/kb" \
        -p "${API_PORT}:8083" \
        -p "${METRICS_PORT}:9093" \
        -e "INPUT_DIR=/work/input" \
        -e "OUTPUT_DIR=/work/output" \
        -e "WAN2_MODEL=wan2-base" \
        -e "API_PORT=8083" \
        -e "METRICS_PORT=9093" \
        "$IMAGE_NAME" \
        python3 /opt/app/main.py --daemon --log-level INFO

    wait_for_health "$DAEMON_CONTAINER_NAME"
    log_success "Daemon mode started successfully"
    log_info "API available at: http://localhost:${API_PORT}"
    log_info "Metrics available at: http://localhost:${METRICS_PORT}/metrics"
}

# Start watch mode
start_watch() {
    setup_directories
    ensure_image

    if container_running "$CONTAINER_NAME"; then
        log_error "Container $CONTAINER_NAME is already running. Stop it first."
        return 1
    fi

    log_info "Starting WAN2 Agent in watch mode..."
    docker run --rm -it \
        --name "$CONTAINER_NAME" \
        -v "${INPUT_DIR}:/work/input" \
        -v "${OUTPUT_DIR}:/work/output" \
        -v "${APP_DIR}:/opt/app" \
        -v "${KB_DIR}:/work/kb" \
        -p "${API_PORT}:8083" \
        -p "${METRICS_PORT}:9093" \
        -e "INPUT_DIR=/work/input" \
        -e "OUTPUT_DIR=/work/output" \
        -e "WAN2_MODEL=wan2-base" \
        "$IMAGE_NAME" \
        python3 /opt/app/main.py -w -s 5 --log-level INFO
}

# Run batch processing
run_batch() {
    setup_directories
    ensure_image

    log_info "Running batch processing..."
    docker run --rm \
        --name "$CONTAINER_NAME" \
        -v "${INPUT_DIR}:/work/input" \
        -v "${OUTPUT_DIR}:/work/output" \
        -v "${APP_DIR}:/opt/app" \
        -v "${KB_DIR}:/work/kb" \
        -e "INPUT_DIR=/work/input" \
        -e "OUTPUT_DIR=/work/output" \
        -e "WAN2_MODEL=wan2-base" \
        "$IMAGE_NAME" \
        python3 /opt/app/main.py --log-level INFO

    log_success "Batch processing completed"
}

# Check status
check_status() {
    local any_running=false

    echo "=== Container Status ==="

    if container_running "$DAEMON_CONTAINER_NAME"; then
        echo -e "${GREEN}✓${NC} Daemon container ($DAEMON_CONTAINER_NAME) is running"
        any_running=true
    else
        echo -e "${RED}✗${NC} Daemon container ($DAEMON_CONTAINER_NAME) is not running"
    fi

    if container_running "$CONTAINER_NAME"; then
        echo -e "${GREEN}✓${NC} Watch container ($CONTAINER_NAME) is running"
        any_running=true
    else
        echo -e "${RED}✗${NC} Watch container ($CONTAINER_NAME) is not running"
    fi

    echo ""
    echo "=== Service Status ==="

    if $any_running; then
        if curl -s "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} API endpoint (port ${API_PORT}) is responding"
        else
            echo -e "${RED}✗${NC} API endpoint (port ${API_PORT}) is not responding"
        fi

        if curl -s "http://localhost:${METRICS_PORT}/metrics" >/dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} Metrics endpoint (port ${METRICS_PORT}) is responding"
        else
            echo -e "${RED}✗${NC} Metrics endpoint (port ${METRICS_PORT}) is not responding"
        fi
    else
        echo -e "${YELLOW}!${NC} No containers running - cannot check service endpoints"
    fi

    echo ""
    echo "=== Directory Status ==="
    echo "Input directory: $INPUT_DIR ($(ls "$INPUT_DIR" 2>/dev/null | wc -l) files)"
    echo "Output directory: $OUTPUT_DIR ($(ls "$OUTPUT_DIR" 2>/dev/null | wc -l) files)"
    echo "App directory: $APP_DIR"
    echo "KB directory: $KB_DIR"
}

# Check health endpoints
check_health() {
    log_info "Checking health endpoints..."

    echo "=== Health Check ==="
    if curl -s "http://localhost:${API_PORT}/health" | python3 -m json.tool 2>/dev/null; then
        log_success "Health endpoint is responding"
    else
        log_error "Health endpoint is not responding"
        return 1
    fi

    echo ""
    echo "=== Status Check ==="
    if curl -s "http://localhost:${API_PORT}/status" | python3 -m json.tool 2>/dev/null; then
        log_success "Status endpoint is responding"
    else
        log_error "Status endpoint is not responding"
        return 1
    fi

    echo ""
    echo "=== Metrics Sample ==="
    if curl -s "http://localhost:${METRICS_PORT}/metrics" | head -10; then
        log_success "Metrics endpoint is responding"
    else
        log_error "Metrics endpoint is not responding"
        return 1
    fi
}

# Show logs
show_logs() {
    local follow_flag=""
    if [[ "${1:-}" == "-f" ]]; then
        follow_flag="-f"
        log_info "Following container logs (Ctrl-C to stop)..."
    else
        log_info "Showing container logs..."
    fi

    if container_running "$DAEMON_CONTAINER_NAME"; then
        docker logs $follow_flag "$DAEMON_CONTAINER_NAME"
    elif container_running "$CONTAINER_NAME"; then
        docker logs $follow_flag "$CONTAINER_NAME"
    else
        log_error "No running containers found"
        return 1
    fi
}

# Stop containers
stop_containers() {
    local stopped=false

    if container_running "$DAEMON_CONTAINER_NAME"; then
        log_info "Stopping daemon container..."
        docker stop "$DAEMON_CONTAINER_NAME" >/dev/null
        log_success "Daemon container stopped"
        stopped=true
    fi

    if container_running "$CONTAINER_NAME"; then
        log_info "Stopping watch container..."
        docker stop "$CONTAINER_NAME" >/dev/null
        log_success "Watch container stopped"
        stopped=true
    fi

    if ! $stopped; then
        log_warning "No running containers found"
    fi
}

# Restart containers
restart_containers() {
    log_info "Restarting containers..."
    stop_containers
    sleep 2
    start_daemon
}

# Remove containers and cleanup
remove_containers() {
    stop_containers

    if container_exists "$DAEMON_CONTAINER_NAME"; then
        log_info "Removing daemon container..."
        docker rm "$DAEMON_CONTAINER_NAME" >/dev/null
        log_success "Daemon container removed"
    fi

    if container_exists "$CONTAINER_NAME"; then
        log_info "Removing watch container..."
        docker rm "$CONTAINER_NAME" >/dev/null
        log_success "Watch container removed"
    fi
}

# Submit job via API
submit_job() {
    local file_path="${1:-}"
    if [[ -z "$file_path" ]]; then
        log_error "Usage: $SCRIPT_NAME job <file_path>"
        return 1
    fi

    if [[ ! -f "$INPUT_DIR/$file_path" ]] && [[ ! -f "$file_path" ]]; then
        log_error "File not found: $file_path"
        return 1
    fi

    if [[ -f "$INPUT_DIR/$file_path" ]]; then
        local full_path="/work/input/$file_path"
    else
        local full_path="$file_path"
    fi

    log_info "Submitting job for: $full_path"

    curl -s -X POST "http://localhost:${API_PORT}/job" \
        -H "Content-Type: application/json" \
        -d "{\"file_path\": \"$full_path\", \"force\": false}" \
        | python3 -m json.tool 2>/dev/null || {
        log_error "Failed to submit job"
        return 1
    }

    log_success "Job submitted successfully"
}

# Trigger batch processing via API
trigger_batch() {
    log_info "Triggering batch processing via API..."

    curl -s -X POST "http://localhost:${API_PORT}/batch" \
        -H "Content-Type: application/json" \
        -d "{\"force\": false}" \
        | python3 -m json.tool 2>/dev/null || {
        log_error "Failed to trigger batch processing"
        return 1
    }

    log_success "Batch processing triggered successfully"
}

# Show usage
show_usage() {
    cat << EOF
CrewAI WAN2 Agent Container Management Script

Usage: $SCRIPT_NAME <command> [options]

Commands:
  daemon          Start daemon mode (background service)
  watch           Start watch mode (foreground)
  batch           Run one-shot batch processing

  status          Show container and service status
  health          Check health endpoints
  logs            Show container logs
  logs -f         Follow container logs

  stop            Stop running containers
  restart         Restart containers
  remove          Remove containers and cleanup

  job <file>      Submit single file job via API
  trigger-batch   Trigger batch processing via API

Environment Variables:
  IMAGE_NAME           Docker image name (default: rrecktek/crewai-wan2:1.0.0)
  CONTAINER_NAME       Container name (default: wan2-agent)
  API_PORT            API port (default: 8083)
  METRICS_PORT        Metrics port (default: 9093)
  INPUT_DIR           Input directory (default: ./input)
  OUTPUT_DIR          Output directory (default: ./output)
  APP_DIR             App directory (default: ./app)
  KB_DIR              Knowledge base directory (default: ./kb)

Examples:
  $SCRIPT_NAME daemon                    # Start daemon mode
  $SCRIPT_NAME watch                     # Start watch mode
  $SCRIPT_NAME batch                     # Run batch processing
  $SCRIPT_NAME status                    # Check status
  $SCRIPT_NAME logs -f                   # Follow logs
  $SCRIPT_NAME job command.json          # Submit specific file
  $SCRIPT_NAME trigger-batch             # Trigger batch via API

EOF
}

# Main command dispatcher
main() {
    local command="${1:-help}"

    case "$command" in
        daemon)
            start_daemon
            ;;
        watch)
            start_watch
            ;;
        batch)
            run_batch
            ;;
        status)
            check_status
            ;;
        health)
            check_health
            ;;
        logs)
            show_logs "${2:-}"
            ;;
        stop)
            stop_containers
            ;;
        restart)
            restart_containers
            ;;
        remove)
            remove_containers
            ;;
        job)
            submit_job "${2:-}"
            ;;
        trigger-batch)
            trigger_batch
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Ensure script is executable and run main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
