#!/usr/bin/env python3
"""
CAAL Voice Framework - Voice Agent
==================================

A voice assistant with MCP integrations for n8n workflows.

Usage:
    python voice_agent.py dev

Configuration:
    - .env: Environment variables (MCP URL, model settings)
    - prompt/default.md: Agent system prompt

Environment Variables:
    SPEACHES_URL        - Speaches STT service URL (default: "http://speaches:8000")
    KOKORO_URL          - Kokoro TTS service URL (default: "http://kokoro:8880")
    WHISPER_MODEL       - Whisper model for STT (default: "Systran/faster-whisper-small")
    TTS_VOICE           - Kokoro voice name (default: "af_heart")
    OLLAMA_MODEL        - Ollama model name (default: "qwen3:8b")
    OLLAMA_THINK        - Enable thinking mode (default: "false")
    TIMEZONE            - Timezone for date/time (default: "Pacific Time")
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

import requests

# Add src directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv

# Load environment variables from .env
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, ".env"))

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, mcp
from livekit.plugins import silero, openai

from caal import OllamaLLM
from caal.integrations import (
    load_mcp_config,
    initialize_mcp_servers,
    WebSearchTools,
    discover_n8n_workflows,
)
from caal.llm import ollama_llm_node, ToolDataCache
from caal import session_registry

# Configure logging (LiveKit CLI reconfigures root logger, so set our level explicitly)
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

# Suppress verbose logs from dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)  # MCP client SSE/JSON-RPC spam
logging.getLogger("livekit").setLevel(logging.WARNING)  # LiveKit internal logs
logging.getLogger("livekit_api").setLevel(logging.WARNING)  # Rust bridge logs
logging.getLogger("caal").setLevel(logging.INFO)  # Our package - INFO level

# =============================================================================
# Configuration
# =============================================================================

# Infrastructure config (from .env only - URLs, tokens, etc.)
SPEACHES_URL = os.getenv("SPEACHES_URL", "http://speaches:8000")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-small")
KOKORO_URL = os.getenv("KOKORO_URL", "http://kokoro:8880")
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").lower() == "true"
TIMEZONE_ID = os.getenv("TIMEZONE", "America/Los_Angeles")
TIMEZONE_DISPLAY = os.getenv("TIMEZONE_DISPLAY", "Pacific Time")

# Import settings module for runtime-configurable values
from caal import settings as settings_module


def get_runtime_settings() -> dict:
    """Get runtime-configurable settings.

    These can be changed via the settings UI without rebuilding.
    Falls back to .env values for backwards compatibility.
    """
    settings = settings_module.load_settings()

    return {
        "tts_voice": settings.get("tts_voice") or os.getenv("TTS_VOICE", "am_puck"),
        "model": settings.get("model") or os.getenv("OLLAMA_MODEL", "ministral-3:8b"),
        "temperature": settings.get("temperature", float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))),
        "num_ctx": settings.get("num_ctx", int(os.getenv("OLLAMA_NUM_CTX", "8192"))),
        "max_turns": settings.get("max_turns", int(os.getenv("OLLAMA_MAX_TURNS", "20"))),
        "tool_cache_size": settings.get("tool_cache_size", int(os.getenv("TOOL_CACHE_SIZE", "3"))),
    }


def load_prompt() -> str:
    """Load and populate prompt template with date context."""
    return settings_module.load_prompt_with_context(
        timezone_id=TIMEZONE_ID,
        timezone_display=TIMEZONE_DISPLAY,
    )


# =============================================================================
# Agent Definition
# =============================================================================

# Type alias for tool status callback
ToolStatusCallback = callable  # async (bool, list[str], list[dict]) -> None


class VoiceAssistant(WebSearchTools, Agent):
    """Voice assistant with MCP tools and web search."""

    def __init__(
        self,
        ollama_llm: OllamaLLM,
        mcp_servers: dict[str, mcp.MCPServerHTTP] | None = None,
        n8n_workflow_tools: list[dict] | None = None,
        n8n_workflow_name_map: dict[str, str] | None = None,
        n8n_base_url: str | None = None,
        on_tool_status: ToolStatusCallback | None = None,
        tool_cache_size: int = 3,
        max_turns: int = 20,
    ) -> None:
        super().__init__(
            instructions=load_prompt(),
            llm=ollama_llm,  # Satisfies LLM interface requirement
        )

        # All MCP servers (for multi-MCP support)
        # Named _caal_mcp_servers to avoid conflict with LiveKit's internal _mcp_servers handling
        self._caal_mcp_servers = mcp_servers or {}

        # n8n-specific for workflow execution (n8n uses webhook-based execution)
        self._n8n_workflow_tools = n8n_workflow_tools or []
        self._n8n_workflow_name_map = n8n_workflow_name_map or {}
        self._n8n_base_url = n8n_base_url

        # Callback for publishing tool status to frontend
        self._on_tool_status = on_tool_status

        # Context management: tool data cache and sliding window
        self._tool_data_cache = ToolDataCache(max_entries=tool_cache_size)
        self._max_turns = max_turns

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Custom LLM node using Ollama with think parameter for low latency."""
        # Access config from OllamaLLM instance via self.llm
        async for chunk in ollama_llm_node(
            self,
            chat_ctx,
            model=self.llm.model,
            think=self.llm.think,
            temperature=self.llm.temperature,
            num_ctx=self.llm.num_ctx,
            tool_data_cache=self._tool_data_cache,
            max_turns=self._max_turns,
        ):
            yield chunk


# =============================================================================
# Agent Entrypoint
# =============================================================================

async def entrypoint(ctx: agents.JobContext) -> None:
    """Main entrypoint for the voice agent."""

    # Start webhook server in the same event loop (first job only)
    global _webhook_server_task
    if _webhook_server_task is None:
        _webhook_server_task = asyncio.create_task(start_webhook_server())

    logger.debug(f"Joining room: {ctx.room.name}")
    await ctx.connect()

    # Load MCP servers from config
    mcp_servers = {}
    try:
        mcp_configs = load_mcp_config()
        mcp_servers = await initialize_mcp_servers(mcp_configs)
    except Exception as e:
        logger.warning(f"Failed to load MCP config: {e}")

    # Discover n8n workflows (n8n uses webhook-based execution, not MCP tools)
    n8n_workflow_tools = []
    n8n_workflow_name_map = {}
    n8n_base_url = None
    n8n_mcp = mcp_servers.get("n8n")
    if n8n_mcp:
        try:
            # Extract base URL from n8n MCP server config
            n8n_config = next((c for c in mcp_configs if c.name == "n8n"), None)
            if n8n_config:
                # URL format: http://HOST:PORT/mcp-server/http
                # Base URL: http://HOST:PORT
                url_parts = n8n_config.url.rsplit("/", 2)
                n8n_base_url = url_parts[0] if len(url_parts) >= 2 else n8n_config.url

            n8n_workflow_tools, n8n_workflow_name_map = await discover_n8n_workflows(
                n8n_mcp, n8n_base_url
            )
        except Exception as e:
            logger.warning(f"Failed to discover n8n workflows: {e}")

    # Get runtime settings (from settings.json with .env fallback)
    runtime = get_runtime_settings()

    # Create OllamaLLM instance (config lives here, accessed via self.llm in agent)
    ollama_llm = OllamaLLM(
        model=runtime["model"],
        think=OLLAMA_THINK,
        temperature=runtime["temperature"],
        num_ctx=runtime["num_ctx"],
    )

    # Log configuration
    logger.info("=" * 60)
    logger.info("STARTING VOICE AGENT")
    logger.info("=" * 60)
    logger.info(f"  STT: {SPEACHES_URL} ({WHISPER_MODEL})")
    logger.info(f"  TTS: {KOKORO_URL} ({runtime['tts_voice']})")
    logger.info(
        f"  LLM: Ollama ({runtime['model']}, think={OLLAMA_THINK}, num_ctx={runtime['num_ctx']})"
    )
    logger.info(f"  MCP: {list(mcp_servers.keys()) or 'None'}")
    logger.info("=" * 60)

    # Create session with Speaches STT and Kokoro TTS (both OpenAI-compatible)
    session = AgentSession(
        stt=openai.STT(
            base_url=f"{SPEACHES_URL}/v1",
            api_key="not-needed",  # Speaches doesn't require auth
            model=WHISPER_MODEL,
        ),
        llm=ollama_llm,
        tts=openai.TTS(
            base_url=f"{KOKORO_URL}/v1",
            api_key="not-needed",  # Kokoro doesn't require auth
            model="kokoro",
            voice=runtime["tts_voice"],
        ),
        vad=silero.VAD.load(),
    )

    # ==========================================================================
    # Round-trip latency tracking
    # ==========================================================================

    _transcription_time: float | None = None

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev) -> None:
        nonlocal _transcription_time
        _transcription_time = time.perf_counter()
        logger.debug(f"User said: {ev.transcript[:80]}...")

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev) -> None:
        nonlocal _transcription_time
        if ev.new_state == "speaking" and _transcription_time is not None:
            latency_ms = (time.perf_counter() - _transcription_time) * 1000
            logger.info(f"ROUND-TRIP LATENCY: {latency_ms:.0f}ms (LLM + TTS)")
            _transcription_time = None

    async def _publish_tool_status(
        tool_used: bool,
        tool_names: list[str],
        tool_params: list[dict],
    ) -> None:
        """Publish tool usage status to frontend via data packet."""
        import json
        payload = json.dumps({
            "tool_used": tool_used,
            "tool_names": tool_names,
            "tool_params": tool_params,
        })

        try:
            await ctx.room.local_participant.publish_data(
                payload.encode("utf-8"),
                reliable=True,
                topic="tool_status",
            )
            logger.debug(f"Published tool status: used={tool_used}, names={tool_names}")
        except Exception as e:
            logger.warning(f"Failed to publish tool status: {e}")

    # ==========================================================================

    # Create agent with OllamaLLM and all MCP servers
    assistant = VoiceAssistant(
        ollama_llm=ollama_llm,
        mcp_servers=mcp_servers,
        n8n_workflow_tools=n8n_workflow_tools,
        n8n_workflow_name_map=n8n_workflow_name_map,
        n8n_base_url=n8n_base_url,
        on_tool_status=_publish_tool_status,
        tool_cache_size=runtime["tool_cache_size"],
        max_turns=runtime["max_turns"],
    )

    # Start session
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(),
    )

    # Register session for webhook access
    session_registry.register(ctx.room.name, session, assistant)

    # Create event to wait for session close
    close_event = asyncio.Event()

    @session.on("close")
    def on_session_close(ev) -> None:
        logger.info(f"Session closed: {ev.reason}")
        close_event.set()

    try:
        # Send initial greeting
        await session.generate_reply(
            instructions="Greet the user briefly and let them know you're ready to help."
        )

        logger.info("Agent ready - listening for speech...")

        # Wait until session closes (room disconnects, etc.)
        await close_event.wait()

    finally:
        # Unregister session on cleanup
        session_registry.unregister(ctx.room.name)


# =============================================================================
# Model Preloading
# =============================================================================


def preload_models():
    """Preload STT and LLM models on startup.

    Ensures models are ready before first user connection, avoiding
    delays on first request (especially important on HDDs).

    Note: Kokoro (remsky/kokoro-fastapi) preloads its own models at startup.
    """
    speaches_url = os.getenv("SPEACHES_URL", "http://speaches:8000")
    whisper_model = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-medium")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    ollama_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

    logger.info("Preloading models...")

    # Download Whisper STT model
    try:
        logger.info(f"  Loading STT: {whisper_model}")
        response = requests.post(
            f"{speaches_url}/v1/models/{whisper_model}",
            timeout=300
        )
        if response.status_code == 200:
            logger.info("  ✓ STT ready")
        else:
            logger.warning(f"  STT model download returned {response.status_code}")
    except Exception as e:
        logger.warning(f"  Failed to preload STT model: {e}")

    # Warm up Ollama LLM with correct num_ctx (loads model into VRAM)
    try:
        logger.info(f"  Loading LLM: {ollama_model} (num_ctx={ollama_num_ctx})")
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": ollama_model,
                "prompt": "hi",
                "stream": False,
                "keep_alive": -1,
                "options": {"num_ctx": ollama_num_ctx}
            },
            timeout=180
        )
        if response.status_code == 200:
            logger.info("  ✓ LLM ready")
        else:
            logger.warning(f"  LLM warmup returned {response.status_code}")
    except Exception as e:
        logger.warning(f"  Failed to preload LLM: {e}")


# =============================================================================
# Webhook Server
# =============================================================================

WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8889"))

# Global reference to webhook server task (started in entrypoint)
_webhook_server_task: asyncio.Task | None = None


async def start_webhook_server():
    """Start FastAPI webhook server in the current event loop.

    This runs the webhook server in the same event loop as the LiveKit agent,
    avoiding cross-thread async issues that cause 200x slower MCP calls.
    """
    import uvicorn
    from caal.webhooks import app

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=WEBHOOK_PORT,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    logger.debug(f"Starting webhook server on port {WEBHOOK_PORT}")
    await server.serve()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Preload models before starting worker
    preload_models()

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Suppress memory warnings (models use ~1GB, this is expected)
            job_memory_warn_mb=0,
        )
    )
