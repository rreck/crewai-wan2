# CrewAI WAN2 Agent Makefile
# Copyright (c) RRECKTEK LLC

.PHONY: help build daemon watch batch status health logs clean test

IMAGE_NAME ?= rrecktek/crewai-wan2:1.0.0
CONTAINER_NAME ?= wan2-agent-daemon

help:
	@echo "CrewAI WAN2 Agent - Make Commands"
	@echo ""
	@echo "Build & Setup:"
	@echo "  make build          Build Docker image"
	@echo "  make quickstart     Build and start daemon mode"
	@echo ""
	@echo "Operation Modes:"
	@echo "  make daemon         Start background service"
	@echo "  make watch          Start foreground watch mode"
	@echo "  make batch          Run one-time batch processing"
	@echo ""
	@echo "Management:"
	@echo "  make stop           Stop running container"
	@echo "  make restart        Restart container"
	@echo "  make status         Show container status"
	@echo "  make health         Check API health"
	@echo ""
	@echo "Monitoring:"
	@echo "  make logs           Show container logs"
	@echo "  make logs-f         Follow container logs"
	@echo "  make metrics        Show Prometheus metrics"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run test video generation"
	@echo "  make test-butterfly Test butterfly video"
	@echo "  make api-test       Test all API endpoints"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove containers"
	@echo "  make clean-all      Remove containers and images"

build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) .

quickstart: build daemon
	@echo "Quickstart complete!"

daemon:
	@./run-wan2-agent-watch.sh daemon

watch:
	@./run-wan2-agent-watch.sh watch

batch:
	@./run-wan2-agent-watch.sh batch

stop:
	@./run-wan2-agent-watch.sh stop

restart:
	@./run-wan2-agent-watch.sh restart

status:
	@./run-wan2-agent-watch.sh status

health:
	@./run-wan2-agent-watch.sh health

logs:
	@./run-wan2-agent-watch.sh logs

logs-f:
	@./run-wan2-agent-watch.sh logs -f

metrics:
	@echo "=== Prometheus Metrics ==="
	@curl -s http://localhost:9093/metrics | grep "wan2_"

test:
	@echo "Running test video generation..."
	@./run-wan2-agent-watch.sh job example_text_to_video.json

test-butterfly:
	@echo "Generating butterfly video..."
	@./run-wan2-agent-watch.sh job test_butterfly.json

api-test:
	@echo "=== Testing API Endpoints ==="
	@echo ""
	@echo "1. Health Check:"
	@curl -s http://localhost:8083/health | python3 -m json.tool
	@echo ""
	@echo "2. Status:"
	@curl -s http://localhost:8083/status | python3 -m json.tool
	@echo ""
	@echo "3. Config:"
	@curl -s http://localhost:8083/config | python3 -m json.tool
	@echo ""
	@echo "API tests complete!"

clean:
	@./run-wan2-agent-watch.sh remove
	@echo "Cleaned up containers"

clean-all: clean
	@docker rmi $(IMAGE_NAME) 2>/dev/null || true
	@echo "Cleaned up containers and images"

.DEFAULT_GOAL := help
