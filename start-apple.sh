#!/bin/bash
# CAAL Startup Script for Apple Silicon
# Usage: ./start-apple.sh [--stop]

set -e
cd "$(dirname "$0")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log() { echo -e "${GREEN}[CAAL]${NC} $1"; }
warn() { echo -e "${YELLOW}[CAAL]${NC} $1"; }
error() { echo -e "${RED}[CAAL]${NC} $1"; }

MLX_PID_FILE="/tmp/caal-mlx-audio.pid"
MLX_LOG_FILE="/tmp/caal-mlx-audio.log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MLX_VENV="$SCRIPT_DIR/.mlx-audio-venv"
MLX_PYTHON="$MLX_VENV/bin/python"

# Load .env file if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
fi

# Set Docker profile based on HTTPS_DOMAIN
if [ -n "${HTTPS_DOMAIN}" ]; then
    DOCKER_PROFILE="--profile https"
else
    DOCKER_PROFILE=""
fi

banner() {
    echo -e "${CYAN}${BOLD}"
    cat << 'EOF'
    ███╗   ███╗██╗     ██╗  ██╗       ██████╗ █████╗  █████╗ ██╗
    ████╗ ████║██║     ╚██╗██╔╝      ██╔════╝██╔══██╗██╔══██╗██║
    ██╔████╔██║██║      ╚███╔╝ █████╗██║     ███████║███████║██║
    ██║╚██╔╝██║██║      ██╔██╗ ╚════╝██║     ██╔══██║██╔══██║██║
    ██║ ╚═╝ ██║███████╗██╔╝ ██╗      ╚██████╗██║  ██║██║  ██║███████╗
    ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝       ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
EOF
    echo -e "${NC}"
    echo -e "  ${BOLD}Voice Assistant for Apple Silicon${NC}"
    echo ""
}

# Load a model with progress feedback
load_model() {
    local model="$1"
    local name="$2"

    curl -s -X POST "http://localhost:8001/v1/models?model_name=$model" > /dev/null &
    local pid=$!

    local spin=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
    local i=0

    while kill -0 $pid 2>/dev/null; do
        printf "\r${GREEN}[CAAL]${NC} Loading %s ${CYAN}%s${NC} " "$name" "${spin[$i]}"
        i=$(( (i + 1) % 10 ))
        sleep 0.1
    done

    wait $pid
    printf "\r${GREEN}[CAAL]${NC} ✓ %s loaded                    \n" "$name"
}

stop_all() {
    echo ""
    log "Stopping CAAL..."

    # Stop Docker
    log "Stopping Docker containers..."
    docker compose -f docker-compose.apple.yaml $DOCKER_PROFILE down || true

    # Stop mlx-audio
    if [ -f "$MLX_PID_FILE" ]; then
        PID=$(cat "$MLX_PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            log "mlx-audio stopped (PID $PID)"
        fi
        rm -f "$MLX_PID_FILE"
    fi

    echo ""
    log "CAAL stopped."
    exit 0
}

# Handle --stop flag
if [ "$1" == "--stop" ]; then
    stop_all
fi

setup_mlx_audio() {
    log "Setting up mlx-audio environment..."

    # Check if venv exists but is corrupted (missing pip - can happen with uv-created venvs)
    if [ -d "$MLX_VENV" ] && ! "$MLX_VENV/bin/python" -c "import pip" 2>/dev/null; then
        warn "Virtual environment is corrupted (missing pip). Recreating..."
        rm -rf "$MLX_VENV"
    fi

    # Create virtual environment if it doesn't exist
    # Use Python 3.11 explicitly for compatibility with mlx-audio dependencies
    if [ ! -d "$MLX_VENV" ]; then
        log "Creating virtual environment at $MLX_VENV..."
        if command -v python3.11 &> /dev/null; then
            python3.11 -m venv "$MLX_VENV"
        else
            warn "Python 3.11 not found, using default python3 (may have compatibility issues)"
            python3 -m venv "$MLX_VENV"
        fi
    fi

    # Verify pip is available
    if ! "$MLX_VENV/bin/python" -c "import pip" 2>/dev/null; then
        error "Failed to create virtual environment with pip. Please check your Python installation."
        exit 1
    fi

    # Upgrade pip
    "$MLX_VENV/bin/pip" install --upgrade pip -q

    # Install mlx-audio and all dependencies
    log "Installing mlx-audio and dependencies (this may take a few minutes)..."

    # Install all mlx-audio dependencies in one command
    "$MLX_VENV/bin/pip" install -q \
        mlx-audio \
        soundfile fastapi uvicorn webrtcvad python-multipart \
        numba tiktoken scipy tqdm \
        loguru misaki num2words spacy phonemizer-fork espeakng-loader torch

    log "✓ mlx-audio environment ready"
    echo ""

    # Pre-download models
    log "Pre-downloading models (first time may take a few minutes)..."
    echo ""

    log "Downloading Whisper STT model..."
    "$MLX_PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/whisper-medium-mlx', local_files_only=False)
print('Done')
" 2>/dev/null || warn "Whisper model will be downloaded on first use"

    log "Downloading Kokoro TTS model..."
    "$MLX_PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('prince-canuma/Kokoro-82M', local_files_only=False)
print('Done')
" 2>/dev/null || warn "Kokoro model will be downloaded on first use"

    echo ""
    log "✓ Models ready"
}

banner
log "Starting CAAL..."
echo ""

# Check Ollama
printf "${GREEN}[CAAL]${NC} Checking Ollama... "
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo ""
    error "Ollama is not accessible on localhost:11434"
    error "Run: ollama serve"
    exit 1
fi
echo -e "${GREEN}✓${NC}"

# Check/setup mlx-audio environment
# Verify: python exists, pip is available, and mlx_audio can be imported
if [ ! -f "$MLX_PYTHON" ] || \
   ! "$MLX_PYTHON" -c "import pip" 2>/dev/null || \
   ! "$MLX_PYTHON" -c "import mlx_audio" 2>/dev/null; then
    setup_mlx_audio
else
    printf "${GREEN}[CAAL]${NC} Checking mlx-audio... "
    echo -e "${GREEN}✓${NC}"
fi

# Check if mlx-audio is already running
MLX_RUNNING=false
if [ -f "$MLX_PID_FILE" ]; then
    PID=$(cat "$MLX_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        MLX_RUNNING=true
        log "✓ mlx-audio already running (PID $PID)"
    fi
fi

# Start mlx-audio if not running
if [ "$MLX_RUNNING" = false ]; then
    printf "${GREEN}[CAAL]${NC} Starting mlx-audio server... "

    # Start in background using dedicated venv
    nohup "$MLX_PYTHON" -m mlx_audio.server --host 0.0.0.0 --port 8001 > "$MLX_LOG_FILE" 2>&1 &
    MLX_PID=$!
    echo $MLX_PID > "$MLX_PID_FILE"

    # Wait for server to be ready
    for i in {1..30}; do
        if curl -s http://localhost:8001/docs > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if ! curl -s http://localhost:8001/docs > /dev/null 2>&1; then
        echo ""
        error "mlx-audio failed to start. Logs: $MLX_LOG_FILE"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} (PID $MLX_PID)"
    echo ""

    # Preload models with progress
    log "Preloading models (first time may take a few minutes)..."
    echo ""

    load_model "mlx-community/whisper-medium-mlx" "Whisper STT"
    load_model "prince-canuma/Kokoro-82M" "Kokoro TTS"

    echo ""
fi

# Start Docker services
log "Starting Docker services..."
docker compose -f docker-compose.apple.yaml $DOCKER_PROFILE up -d

# Wait for services
printf "${GREEN}[CAAL]${NC} Waiting for services"
for i in {1..5}; do
    printf "."
    sleep 1
done
echo -e " ${GREEN}✓${NC}"

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  CAAL is ready!${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}Web interface:${NC}  http://localhost:3000"
echo -e "  ${BOLD}Stop command:${NC}   ./start-apple.sh --stop"
echo ""
echo -e "  ${BOLD}Logs:${NC}"
echo -e "    mlx-audio:  tail -f $MLX_LOG_FILE"
echo -e "    agent:      docker compose -f docker-compose.apple.yaml logs -f agent"
echo ""
