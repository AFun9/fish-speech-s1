#!/bin/bash
# Multi-instance TTS: independent single-worker processes per GPU
# Usage: bash launch_multi.sh [instances_per_gpu] [--compile]

set -e

INSTANCES_PER_GPU=${1:-7}
COMPILE_FLAG=""
if [[ "$*" == *"--compile"* ]]; then
    COMPILE_FLAG="--compile"
fi

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

CONDA_BASE=$(conda info --base 2>/dev/null)
PYTHON="$CONDA_BASE/envs/fish-s1/bin/python"

LB_PORT=8080
GPU0_BASE_PORT=8081   # GPU0: 8081, 8082, ...
GPU1_BASE_PORT=8181   # GPU1: 8181, 8182, ...

TOTAL=$((INSTANCES_PER_GPU * 2))
echo "============================================"
echo "  Multi-Instance TTS Server"
echo "  Instances per GPU: $INSTANCES_PER_GPU"
echo "  Total instances: $TOTAL"
echo "  Compile: ${COMPILE_FLAG:-disabled}"
echo "  GPU0 ports: $GPU0_BASE_PORT - $((GPU0_BASE_PORT + INSTANCES_PER_GPU - 1))"
echo "  GPU1 ports: $GPU1_BASE_PORT - $((GPU1_BASE_PORT + INSTANCES_PER_GPU - 1))"
echo "  LB: :$LB_PORT"
echo "============================================"

PIDS=()
ALL_BACKENDS=""

cleanup() {
    echo ""
    echo "Shutting down ${#PIDS[@]} processes..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    sleep 2
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    echo "Done."
    exit 0
}
trap cleanup INT TERM

cd "$PROJECT_DIR"

# Launch instances
for gpu in 0 1; do
    if [ "$gpu" -eq 0 ]; then BASE_PORT=$GPU0_BASE_PORT; else BASE_PORT=$GPU1_BASE_PORT; fi

    for i in $(seq 0 $((INSTANCES_PER_GPU - 1))); do
        PORT=$((BASE_PORT + i))
        echo "  Starting GPU$gpu instance on :$PORT ..."
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON -m tools.api_server \
            --listen "0.0.0.0:$PORT" --workers 1 $COMPILE_FLAG \
            > "$LOG_DIR/gpu${gpu}_${i}.log" 2>&1 &
        PIDS+=($!)

        if [ -z "$ALL_BACKENDS" ]; then
            ALL_BACKENDS="127.0.0.1:$PORT"
        else
            ALL_BACKENDS="$ALL_BACKENDS,127.0.0.1:$PORT"
        fi
    done
done

echo ""
echo "Waiting for all $TOTAL instances to load (this takes a while)..."

for gpu in 0 1; do
    if [ "$gpu" -eq 0 ]; then BASE_PORT=$GPU0_BASE_PORT; else BASE_PORT=$GPU1_BASE_PORT; fi

    for i in $(seq 0 $((INSTANCES_PER_GPU - 1))); do
        PORT=$((BASE_PORT + i))
        attempt=0
        while ! curl -sf http://127.0.0.1:$PORT/v1/health > /dev/null 2>&1; do
            attempt=$((attempt + 1))
            if [ $attempt -gt 600 ]; then
                echo "  TIMEOUT: GPU$gpu :$PORT"
                tail -5 "$LOG_DIR/gpu${gpu}_${i}.log"
                cleanup
            fi
            sleep 1
            if [ $((attempt % 30)) -eq 0 ]; then
                echo "  Still loading GPU$gpu :$PORT ... (${attempt}s)"
            fi
        done
        echo "  GPU$gpu :$PORT ready"
    done
done

# Load balancer
echo ""
echo "Starting load balancer on :$LB_PORT ..."
$PYTHON "$PROJECT_DIR/lb_proxy.py" \
    --listen-port "$LB_PORT" \
    --backends "$ALL_BACKENDS" \
    > "$LOG_DIR/lb.log" 2>&1 &
PIDS+=($!)

echo ""
echo "============================================"
echo "  ALL $TOTAL INSTANCES READY"
echo "  Endpoint: http://0.0.0.0:$LB_PORT/v1/tts"
echo "  Backends: $ALL_BACKENDS"
echo "============================================"
echo "Ctrl+C to stop. Logs in $LOG_DIR/"

wait
