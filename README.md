# CAAL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LiveKit](https://img.shields.io/badge/LiveKit-Agents-purple.svg)](https://docs.livekit.io/agents/)

> **Self-hosted voice assistant you actually own.  Secure by design - the LLM never sees your API keys.**

CAAL is an open-source voice assistant built on [LiveKit Agents](https://docs.livekit.io/agents/) that runs entirely on your hardware. Your voice, your data, your credentials — all on your network.

![CAAL Voice Assistant](frontend/.github/assets/readme-hero.webp)

---

### Why CAAL?

**Secure by architecture.** The model never sees your API keys. Ever. Credentials live in [n8n's](https://n8n.io/) encrypted credential store. The LLM sends parameters to a webhook, the workflow handles auth. Even if a prompt injection succeeds, the model can only call pre-built workflows — no shell access, no curl, no ability to transmit data. It's an air gap for the LLM. Every tool in the [registry](https://github.com/CoreWorxLab/caal-tools) goes through automated security review and human approval before it's available to install.

**Purpose-built model.** CAAL ships with [`caal-ministral`](https://ollama.com/coreworxlab/caal-ministral) — a fine-tuned 8B model trained specifically for voice tool calling. It knows how to control your smart home, chain tools together, and respond naturally. The LLM is one piece of the architecture, not the architecture. It handles decisions that code can't — which tool to call, what parameters to use, when to chain steps. Everything else is code. That's why an 8B model works.
```bash
ollama pull coreworxlab/caal-qwen3.5-9b
```

**Infinitely extensible.** Any [n8n](https://n8n.io/) workflow becomes a voice-activated tool. Control [Home Assistant](https://www.home-assistant.io/) devices, query APIs, automate your life — then share your tools with the community via the [CAAL Tool Registry](https://github.com/CoreWorxLab/caal-tools). Tools follow a suite convention — fewer tools, better accuracy, more reliable routing.

**Local by default.** Runs fully on your network with [Ollama](https://ollama.ai/). No accounts, no telemetry, no cloud dependency. Want to use [Groq](https://groq.com/), [OpenRouter](https://openrouter.ai/), [Requesty](https://requesty.ai/), or any OpenAI-compatible API? Your choice. Your credentials and tool executions never leave your network regardless.

---

## Features

- **Tool Registry** — Browse and install community tools with one click. Every submission goes through automated security review and human approval
- **Tool Chaining** — Sequential multi-tool calls in one prompt. The model uses real data from each step to inform the next
- **Home Assistant** — Voice control across lights, covers, locks, climate, media, and more via `hass`
- **n8n Workflows** — Any workflow becomes a tool. Visual, inspectable, shareable, auditable through n8n's execution history
- **Flexible Providers** — Ollama, Groq, OpenRouter, Requesty, or any OpenAI-compatible API. Speaches or Groq for STT. Kokoro or Piper for TTS
- **Short-Term Memory** — Store and recall information across sessions
- **Internationalization** — English, French, Italian, with more coming
- **Wake Word** — "Hey Cal" via OpenWakeWord
- **Web Search** — DuckDuckGo integration for real-time information
- **Mobile App** — Android client available from [Releases](https://github.com/CoreWorxLab/caal/releases)
- **Webhook API** — REST API for announcements, settings, and external triggers

---

## Quick Start

```bash
git clone https://github.com/CoreWorxLab/caal.git
cd caal
cp .env.example .env
nano .env  # Set CAAL_HOST_IP to your server's LAN IP

# If using Ollama (recommended)
ollama pull coreworxlab/caal-ministral:latest

docker compose up -d
```

Open `https://YOUR_SERVER_IP:3443` and complete the setup wizard.

> Requires Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and 8GB+ VRAM. For CPU-only or Apple Silicon setups, see [Deployment Options](#deployment-options).

## Deployment Options

| Mode | Hardware | Command | Guide |
|------|----------|---------|-------|
| **GPU** | Linux + NVIDIA GPU | `docker compose up -d` | Quick Start above |
| **CPU-only** | Any Docker host | `docker compose -f docker-compose.cpu.yaml up -d` | [Wiki: CPU Mode](https://github.com/CoreWorxLab/CAAL/wiki) |
| **Apple Silicon** | M1/M2/M3/M4 Mac | `./start-apple.sh` | [Apple Silicon Guide](docs/APPLE-SILICON.md) |
| **Distributed** | GPU server + frontend | See guide | [Distributed Guide](docs/DISTRIBUTED-DEPLOYMENT.md) |

### HTTPS & Network

HTTPS is enabled by default with auto-generated self-signed certificates. This is required because browsers block microphone access on non-`localhost` HTTP.

To avoid the browser certificate warning, trust the auto-generated cert:
```bash
./trust-cert.sh
```
This works on macOS and Linux (Debian/Ubuntu, RHEL/Fedora, Arch, Chrome, Firefox). Pass `--yes` to skip the confirmation prompt. On Apple Silicon, `start-apple.sh` runs it automatically.

Alternatively, for browser-trusted certs use [mkcert](https://github.com/FiloSottile/mkcert):
```bash
mkcert -install && mkcert 192.168.1.100
mkdir -p certs && mv 192.168.1.100.pem certs/server.crt && mv 192.168.1.100-key.pem certs/server.key
```

For remote access via [Tailscale](https://tailscale.com/), set `HTTPS_DOMAIN` in `.env` to your Tailscale domain. See the [wiki](https://github.com/CoreWorxLab/CAAL/wiki) for details.

> The mobile app connects via LiveKit directly and doesn't require HTTPS.

---

## Documentation

| Resource | Description |
|----------|-------------|
| **[Wiki](https://github.com/CoreWorxLab/CAAL/wiki)** | Full documentation — architecture, configuration, deployment |
| [Home Assistant](docs/HOME-ASSISTANT.md) | Smart home integration setup and usage |
| [n8n Workflows](docs/N8N-WORKFLOWS.md) | Creating and connecting workflow tools |
| [Apple Silicon](docs/APPLE-SILICON.md) | Running on M1/M2/M3/M4 Macs |
| [Distributed Deployment](docs/DISTRIBUTED-DEPLOYMENT.md) | Split GPU backend and frontend |
| [Internationalization](docs/I18N.md) | Adding language support |
| [Tool Registry](https://github.com/CoreWorxLab/caal-tools) | Browse and submit community tools |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and contribution guidelines |

---

## Architecture

```
                          https://<IP>:3443    https://<IP>:7443
                                │                     │
                          ┌─────┴─────────────────────┴─────┐
                          │           nginx (TLS)           │
                          │         :3443  :7443            │
                          └─────┬─────────────────┬─────────┘
                                │                 │
┌───────────────────────────────┼─────────────────┼───────────────────┐
│  Docker Compose Stack         │                 │                   │
│                               │                 │                   │
│  ┌────────────┐         ┌─────┴──────┐  ┌───────┴────┐  ┌────────┐  │
│  │  Speaches  │         │  Frontend  │  │  LiveKit   │  │ Kokoro │  │
│  │(STT, GPU)  │         │  (Next.js) │  │   Server   │  │ (TTS)  │  │
│  │   :8000    │         │   :3000    │  │   :7880    │  │ :8880  │  │
│  └─────┬──────┘         └─────┬──────┘  └─────┬──────┘  └───┬────┘  │
│        │                      │               │             │       │
│        └──────────────────────┼───────────────┼─────────────┘       │
│                               │               │                     │
│                         ┌─────┴───────────────┴─────┐               │
│                         │         Agent             │               │
│                         │    (Voice Pipeline)       │               │
│                         │    :8889 (webhooks)       │               │
│                         └─────────┬─────────────────┘               │
│                                   │                                 │
└───────────────────────────────────┼─────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
     ┌─────┴─────┐           ┌──────┴──────┐          ┌──────┴──────┐
     │   Ollama  │           │     n8n     │          │    Home     │
     │Groq / OR  │           │  Workflows  │          │  Assistant  │
     └───────────┘           └─────────────┘          └─────────────┘
                      External Services (via MCP)
```

---

## Community

CAAL is built in the open. If you build a tool, we see the PR. If you find a bug, we see the issue.

- **[Tool Registry](https://github.com/CoreWorxLab/caal-tools)** — Browse, install, and share tools
- **[Discussions](https://github.com/CoreWorxLab/CAAL/discussions)** — Feature requests, questions, feedback
- **[Fine-tuned Model](https://ollama.com/coreworxlab/caal-ministral)** — `ollama pull coreworxlab/caal-ministral`

---

## Related Projects

- [LiveKit Agents](https://github.com/livekit/agents) - Voice agent framework
- [Speaches](https://github.com/speaches-ai/speaches) - Faster-Whisper STT + Piper TTS
- [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) - Kokoro TTS server
- [Ollama](https://ollama.ai/) - Local LLM server
- [Groq](https://groq.com/) - Fast cloud LLM inference
- [OpenRouter](https://openrouter.ai/) - Unified API for 200+ models
- [Requesty](https://requesty.ai/) - OpenAI-compatible LLM gateway
- [n8n](https://n8n.io/) - Workflow automation
- [Home Assistant](https://www.home-assistant.io/) - Smart home platform

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.
