# CAAL on Apple Silicon (M1/M2/M3/M4)

This guide covers running CAAL on Apple Silicon Macs using Metal-accelerated STT/TTS via [mlx-audio](https://github.com/Blaizzy/mlx-audio).

## Why a Separate Setup?

Docker on macOS cannot access the Metal GPU. To get hardware acceleration for Speech-to-Text and Text-to-Speech, these services must run natively on the host while the rest of the stack runs in Docker.

```
┌─────────────────────────────────────────────────────────┐
│  macOS Host                                             │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐                       │
│  │  mlx-audio  │  │   Ollama    │                       │
│  │ (STT + TTS) │  │   (LLM)     │                       │
│  │   :8001     │  │  :11434     │                       │
│  │  [Metal]    │  │   [MPS]     │                       │
│  └──────┬──────┘  └──────┬──────┘                       │
│         │                │                              │
│  ┌──────┴────────────────┴──────────────────────────┐   │
│  │           Docker (ARM64)                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │   │
│  │  │ Frontend │  │ LiveKit  │  │  Agent   │        │   │
│  │  │  :3000   │  │  :7880   │  │  :8889   │        │   │
│  │  └──────────┘  └──────────┘  └──────────┘        │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. Ollama

Install and pull a model:

```bash
brew install ollama
ollama pull ministral-3:8b
ollama serve  # Keep running or use: brew services start ollama
```

> **Note:** `ministral-3:8b` is recommended for its good tool-calling support and low latency. Requires Ollama 0.13.3+.

### 2. mlx-audio

> **Recommended:** The easiest way to install mlx-audio with all its dependencies is to use the `start-apple.sh` script. It will automatically create the virtual environment, install all dependencies, and download the required models. Just run `./start-apple.sh` and everything will be set up for you.

If you prefer manual installation, mlx-audio requires a dedicated virtual environment with all dependencies for STT (Whisper) and TTS (Kokoro).

> **Important:** Use Python 3.11 for compatibility. Python 3.12+ may have issues with some dependencies.

```bash
# Create dedicated virtual environment (Python 3.11 recommended)
python3.11 -m venv ~/.mlx-audio-venv
~/.mlx-audio-venv/bin/pip install --upgrade pip

# Install mlx-audio with all dependencies
~/.mlx-audio-venv/bin/pip install \
    mlx-audio \
    soundfile fastapi uvicorn webrtcvad python-multipart \
    numba tiktoken scipy tqdm \
    loguru misaki num2words spacy phonemizer-fork espeakng-loader torch
```

### 3. Docker Desktop

Download from [docker.com](https://www.docker.com/products/docker-desktop/) and ensure it's running.

### 4. n8n (Optional)

If you want workflow integrations, you need n8n with MCP enabled:
1. Run n8n (Docker or native)
2. Go to **Settings > MCP Access > Enable MCP**
3. Copy the access token

## Installation

### 1. Clone and Configure

```bash
git clone https://github.com/CoreWorxLab/caal.git
cd caal
cp .env.example .env
```

### 2. Edit `.env`

Update these values:

```bash
# Your Mac's LAN IP (find with: ipconfig getifaddr en0)
CAAL_HOST_IP=192.168.1.100

# Ollama (host.docker.internal lets Docker reach your Mac)
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=ministral-3:8b

# mlx-audio runs on port 8001 (8000 is often used by other services)
MLX_AUDIO_URL=http://host.docker.internal:8001

# n8n MCP (optional - if you have n8n running)
N8N_MCP_URL=http://host.docker.internal:5678/mcp-server/http
N8N_MCP_TOKEN=your_token_here

# Suppress Docker warnings
HTTPS_DOMAIN=
```

## Usage

### Start CAAL

```bash
./start-apple.sh
```

This will:
1. Check that Ollama is running
2. Start mlx-audio server (Metal-accelerated)
3. Preload Whisper (STT) and Kokoro (TTS) models
4. Start Docker services (LiveKit, Agent, Frontend)

Output:
```
    ███╗   ███╗██╗     ██╗  ██╗       ██████╗ █████╗  █████╗ ██╗
    ████╗ ████║██║     ╚██╗██╔╝      ██╔════╝██╔══██╗██╔══██╗██║
    ██╔████╔██║██║      ╚███╔╝ █████╗██║     ███████║███████║██║
    ██║╚██╔╝██║██║      ██╔██╗ ╚════╝██║     ██╔══██║██╔══██║██║
    ██║ ╚═╝ ██║███████╗██╔╝ ██╗      ╚██████╗██║  ██║██║  ██║███████╗
    ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝       ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝

  Voice Assistant for Apple Silicon

[CAAL] Starting CAAL...

[CAAL] Checking Ollama... ✓
[CAAL] Starting mlx-audio server... ✓ (PID 12345)

[CAAL] Preloading models (first time may take a few minutes)...

[CAAL] Loading Whisper STT ⠹
[CAAL] ✓ Whisper STT loaded
[CAAL] Loading Kokoro TTS ⠧
[CAAL] ✓ Kokoro TTS loaded

══════════════════════════════════════════════════════════
  CAAL is ready!
══════════════════════════════════════════════════════════

  Web interface:  http://localhost:3000
  Stop command:   ./start-apple.sh --stop
```

### Stop CAAL

```bash
./start-apple.sh --stop
```

### View Logs

```bash
# mlx-audio logs
tail -f /tmp/caal-mlx-audio.log

# Agent logs
docker compose -f docker-compose.apple.yaml logs -f agent
```

## First Run

The first startup takes longer because:
- mlx-audio downloads Whisper (~1.5GB) and Kokoro (~300MB) models
- Docker builds the Agent and Frontend images

Subsequent starts are much faster as everything is cached.

## Troubleshooting

### "Ollama is not accessible"

Make sure Ollama is running:
```bash
ollama serve
# Or check if it's already running:
curl http://localhost:11434/api/tags
```

### "mlx-audio failed to start"

Check the logs for missing dependencies:
```bash
tail -50 /tmp/caal-mlx-audio.log
```

Common missing dependencies:
- `ModuleNotFoundError: No module named 'numba'` → `~/.mlx-audio-venv/bin/pip install numba`
- `ModuleNotFoundError: No module named 'loguru'` → `~/.mlx-audio-venv/bin/pip install loguru`
- `ModuleNotFoundError: No module named 'soundfile'` → `~/.mlx-audio-venv/bin/pip install soundfile`
- `ModuleNotFoundError: No module named 'spacy'` → `~/.mlx-audio-venv/bin/pip install spacy phonemizer-fork`
- `Model type kokoro not supported` → `~/.mlx-audio-venv/bin/pip install loguru misaki num2words spacy phonemizer-fork`
- `Model type whisper not supported` → `~/.mlx-audio-venv/bin/pip install numba`

Check if port 8001 is already in use:
```bash
lsof -i :8001
```

If another service uses it, either stop that service or change the port in both `start-apple.sh` and `.env`.

### Voice not working in browser

On HTTP (not HTTPS), browsers only allow microphone access from `localhost`. Access CAAL at:
- ✅ `http://localhost:3000` (microphone works)
- ❌ `http://192.168.1.100:3000` (text only, no microphone)

For voice from other devices, see the HTTPS setup in the main README.

### Agent can't connect to mlx-audio

Verify Docker can reach your Mac:
```bash
docker run --rm alpine ping -c1 host.docker.internal
```

### Slow first response

Normal - Ollama may need to load the model into memory. Subsequent responses are faster. To keep the model loaded:
```bash
OLLAMA_KEEP_ALIVE=24h ollama serve
```

## Models

### STT (Speech-to-Text)
- Default: `mlx-community/whisper-medium-mlx`
- Alternatives: `whisper-tiny-mlx`, `whisper-small-mlx`, `whisper-large-v3-turbo-mlx`

### TTS (Text-to-Speech)
- Default: `prince-canuma/Kokoro-82M`
- Voice: `af_heart` (configurable in `.env` as `TTS_VOICE`)
- Available voices: `af_heart`, `af_bella`, `af_nova`, `af_alloy` (American female), `bf_emma` (British female)

### LLM
- Default: `ministral-3:8b` (good tool-calling, low latency)
- Alternative: `llama3.1:8b`, `qwen2.5:7b`
- Any Ollama model works, but 7-8B models offer the best latency/quality balance

## Memory Usage

With default models on a 24GB Mac:
- mlx-audio (Whisper + Kokoro): ~2-3GB
- Ollama (ministral-3:8b): ~5GB
- Docker services: ~1GB

Total: ~8-9GB, leaving plenty of headroom.

## Updating

```bash
git pull
docker compose -f docker-compose.apple.yaml build
./start-apple.sh --stop
./start-apple.sh
```
