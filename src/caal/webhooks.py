"""Webhook server for external triggers (announcements, tool reload, wake word, settings).

This module provides HTTP endpoints that allow external systems (like n8n)
and the frontend to trigger actions on the running voice agent.

Endpoints:
    POST /announce           - Make the agent speak a message
    POST /reload-tools       - Refresh MCP tool cache and optionally announce
    POST /wake               - Handle wake word detection (greet user)
    GET  /health             - Health check
    GET  /settings           - Get current settings
    POST /settings           - Update settings
    GET  /prompt             - Get current prompt content
    POST /prompt             - Save custom prompt
    GET  /voices             - List available TTS voices
    GET  /models             - List available LLM models
    GET  /wake-word/status   - Get wake word detection status
    POST /wake-word/enable   - Enable server-side wake word detection
    POST /wake-word/disable  - Disable server-side wake word detection
    GET  /wake-word/models   - List available wake word models

Usage:
    # Start in a background thread from voice_agent.py:
    import threading
    import uvicorn
    from caal.webhooks import app

    def run_webhook_server():
        uvicorn.run(app, host="0.0.0.0", port=8889, log_level="info")

    webhook_thread = threading.Thread(target=run_webhook_server, daemon=True)
    webhook_thread.start()
"""

from __future__ import annotations

import json
import logging
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
from livekit.protocol.models import DataPacket
from livekit.protocol.room import SendDataRequest
from pydantic import BaseModel

from . import registry_cache
from . import settings as settings_module
from .settings import validate_url

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CAAL Webhook API",
    description="External triggers for CAAL voice agent",
    version="1.0.0",
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend can be on different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_livekit_api() -> api.LiveKitAPI:
    """Get a LiveKit API client for sending data messages to agents."""
    return api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL", "http://localhost:7880"),
        api_key=os.getenv("LIVEKIT_API_KEY", "devkey"),
        api_secret=os.getenv("LIVEKIT_API_SECRET", "secret"),
    )


async def send_agent_command(room_name: str, command: dict) -> tuple[bool, str | None]:
    """Send a command to the agent via LiveKit data channel.

    Args:
        room_name: The room to send to
        command: Dict with 'action' and other fields

    Returns:
        Tuple of (success, error_message)
    """
    lk = None
    try:
        lk = get_livekit_api()
        payload = json.dumps(command)
        await lk.room.send_data(
            SendDataRequest(
                room=room_name,
                data=payload.encode("utf-8"),
                kind=DataPacket.Kind.RELIABLE,
                topic="webhook_command",
            )
        )
        return True, None
    except Exception as e:
        error_msg = str(e)
        # Check for common error patterns
        if "no response from servers" in error_msg or "503" in error_msg:
            error_msg = f"No active session in room: {room_name}"
        logger.error(f"Failed to send agent command: {error_msg}")
        return False, error_msg
    finally:
        if lk:
            await lk.aclose()


class AnnounceRequest(BaseModel):
    """Request body for /announce endpoint."""

    message: str
    room_name: str = "voice_assistant_room"


class ReloadToolsRequest(BaseModel):
    """Request body for /reload-tools endpoint."""

    tool_name: str | None = None  # Optional: announce specific tool name
    message: str | None = None  # Optional: custom announcement message (overrides tool_name)
    room_name: str = "voice_assistant_room"


class WakeRequest(BaseModel):
    """Request body for /wake endpoint."""

    room_name: str = "voice_assistant_room"


class WakeResponse(BaseModel):
    """Response body for /wake endpoint."""

    status: str
    room_name: str


class AnnounceResponse(BaseModel):
    """Response body for /announce endpoint."""

    status: str
    room_name: str


class ReloadToolsResponse(BaseModel):
    """Response body for /reload-tools endpoint."""

    status: str
    tool_count: int
    room_name: str


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    active_sessions: list[str]


@app.post("/announce", response_model=AnnounceResponse)
async def announce(req: AnnounceRequest) -> AnnounceResponse:
    """Make the agent speak a message.

    This endpoint sends an announcement command to the agent via LiveKit
    data channel. The agent will speak the provided message using TTS.

    Args:
        req: AnnounceRequest with message and optional room_name

    Returns:
        AnnounceResponse with status

    Raises:
        HTTPException: 404 if no active session, 500 if failed to send
    """
    logger.info(f"Announcing to room {req.room_name}: {req.message[:50]}...")

    success, error = await send_agent_command(
        req.room_name,
        {"action": "announce", "message": req.message},
    )

    if not success:
        status_code = 404 if "No active session" in (error or "") else 500
        raise HTTPException(status_code=status_code, detail=error)

    return AnnounceResponse(status="queued", room_name=req.room_name)


@app.post("/reload-tools", response_model=ReloadToolsResponse)
async def reload_tools(req: ReloadToolsRequest) -> ReloadToolsResponse:
    """Refresh MCP tool cache and optionally announce new tool availability.

    This endpoint sends a reload command to the agent via LiveKit data channel.
    The agent will clear caches, re-discover workflows, and optionally announce:
    - If `message` is provided, speaks that exact message
    - If only `tool_name` is provided, speaks "A new tool called '{tool_name}' is now available."
    - If neither is provided, reloads silently

    Args:
        req: ReloadToolsRequest with optional message, tool_name, and room_name

    Returns:
        ReloadToolsResponse with status

    Raises:
        HTTPException: 500 if failed to send command
    """
    logger.info(f"Reloading tools for room {req.room_name}")

    success, error = await send_agent_command(
        req.room_name,
        {
            "action": "reload_tools",
            "message": req.message,
            "tool_name": req.tool_name,
        },
    )

    if not success:
        status_code = 404 if "No active session" in (error or "") else 500
        raise HTTPException(status_code=status_code, detail=error)

    # Tool count is now handled agent-side, return 0 as placeholder
    return ReloadToolsResponse(
        status="queued",
        tool_count=0,
        room_name=req.room_name,
    )


# =============================================================================
# Registry Cache Endpoints
# =============================================================================


class CacheRegistryEntryRequest(BaseModel):
    """Request body for /cache-registry-entry endpoint."""

    n8n_workflow_id: str
    registry_id: str | None  # None = custom workflow
    version: str | None = None


class CacheRegistryEntryResponse(BaseModel):
    """Response body for /cache-registry-entry endpoint."""

    status: str
    n8n_workflow_id: str


@app.post("/cache-registry-entry", response_model=CacheRegistryEntryResponse)
async def cache_registry_entry(req: CacheRegistryEntryRequest) -> CacheRegistryEntryResponse:
    """Save a registry cache entry for an installed workflow.

    Called by the frontend after successfully installing a tool from the registry.
    Maps the n8n workflow ID to the CAAL registry ID and version.

    Args:
        req: CacheRegistryEntryRequest with workflow ID and registry info

    Returns:
        CacheRegistryEntryResponse with status
    """
    registry_cache.set_cached_entry(
        n8n_workflow_id=req.n8n_workflow_id,
        registry_id=req.registry_id,
        version=req.version,
    )

    return CacheRegistryEntryResponse(
        status="cached",
        n8n_workflow_id=req.n8n_workflow_id,
    )


@app.post("/wake", response_model=WakeResponse)
async def wake(req: WakeRequest) -> WakeResponse:
    """Handle wake word detection - greet the user.

    This endpoint sends a wake command to the agent via LiveKit data channel.
    The agent will play a random greeting from the configured greetings list.

    This is primarily for:
    - Client-side wake word detection (Picovoice - deprecated)
    - Manual testing via curl

    Server-side wake word detection (OpenWakeWord) handles greetings
    directly in voice_agent.py for lower latency.

    Args:
        req: WakeRequest with room_name

    Returns:
        WakeResponse with status

    Raises:
        HTTPException: 500 if failed to send command
    """
    logger.info(f"Wake word detected in room {req.room_name}")

    success, error = await send_agent_command(
        req.room_name,
        {"action": "wake"},
    )

    if not success:
        status_code = 404 if "No active session" in (error or "") else 500
        raise HTTPException(status_code=status_code, detail=error)

    return WakeResponse(status="queued", room_name=req.room_name)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse with status and list of active room names
    """
    try:
        lk = get_livekit_api()
        from livekit.protocol.room import ListRoomsRequest

        response = await lk.room.list_rooms(ListRoomsRequest())
        await lk.aclose()
        rooms = [room.name for room in response.rooms]
    except Exception as e:
        logger.warning(f"Failed to list rooms: {e}")
        rooms = []

    return HealthResponse(
        status="ok",
        active_sessions=rooms,
    )


# =============================================================================
# Settings Endpoints
# =============================================================================


class SettingsResponse(BaseModel):
    """Response body for /settings endpoint."""

    settings: dict
    prompt_content: str
    custom_prompt_exists: bool


class SettingsUpdateRequest(BaseModel):
    """Request body for POST /settings endpoint."""

    settings: dict


class PromptResponse(BaseModel):
    """Response body for /prompt endpoint."""

    prompt: str  # "default" or "custom"
    content: str
    is_custom: bool


class PromptUpdateRequest(BaseModel):
    """Request body for POST /prompt endpoint."""

    content: str


class GreetingsResponse(BaseModel):
    """Response body for /greetings endpoint."""

    language: str
    content: str  # Raw text, one greeting per line


class GreetingsUpdateRequest(BaseModel):
    """Request body for POST /greetings endpoint."""

    language: str
    content: str


class VoicesResponse(BaseModel):
    """Response body for /voices endpoint."""

    voices: list[str]


class ModelsResponse(BaseModel):
    """Response body for /models endpoint."""

    models: list[str]


@app.get("/settings", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    """Get current settings and prompt content.

    Returns:
        SettingsResponse with current settings, prompt content, and custom prompt status.
        Sensitive keys (tokens) are excluded for security.
    """
    settings = settings_module.load_settings_safe()  # Excludes sensitive keys
    language = settings.get("language", "en")
    prompt_content = settings_module.load_prompt_content(language=language)
    custom_exists = settings_module.custom_prompt_exists()

    # Inject greetings from file for backward compat (mobile reads this field)
    settings["wake_greetings"] = settings_module.load_greetings(language)

    return SettingsResponse(
        settings=settings,
        prompt_content=prompt_content,
        custom_prompt_exists=custom_exists,
    )


@app.post("/settings", response_model=SettingsResponse)
async def update_settings(req: SettingsUpdateRequest) -> SettingsResponse:
    """Update settings.

    Args:
        req: SettingsUpdateRequest with settings dict to merge

    Returns:
        SettingsResponse with updated settings
    """
    # Load current settings
    current = settings_module.load_settings()

    # Ignore wake_greetings — now managed via /greetings endpoint (file-based)
    req.settings.pop("wake_greetings", None)

    # Secret fields that should not be overwritten with empty values
    # (UI doesn't show these, so saving would clear them)
    secret_fields = {"groq_api_key", "hass_token", "n8n_token", "n8n_api_key",
                     "openai_api_key", "openrouter_api_key"}

    # Validate URL fields
    url_fields = ["openai_base_url", "ollama_host", "hass_host", "n8n_url"]
    for field in url_fields:
        if field in req.settings and req.settings[field]:
            is_valid, error = validate_url(req.settings[field])
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid {field}: {error}")

    # Merge with new settings (only known keys)
    for key, value in req.settings.items():
        if key in settings_module.DEFAULT_SETTINGS:
            # Don't overwrite secrets with empty values
            if key in secret_fields and not value:
                continue
            current[key] = value

    # Save merged settings
    settings_module.save_settings(current)

    # Reload and return
    settings = settings_module.reload_settings()
    language = settings.get("language", "en")
    prompt_content = settings_module.load_prompt_content(language=language)
    custom_exists = settings_module.custom_prompt_exists()

    # Inject greetings from file for backward compat
    settings["wake_greetings"] = settings_module.load_greetings(language)

    logger.info(f"Settings updated: {list(req.settings.keys())}")

    return SettingsResponse(
        settings=settings,
        prompt_content=prompt_content,
        custom_prompt_exists=custom_exists,
    )


@app.get("/prompt", response_model=PromptResponse)
async def get_prompt() -> PromptResponse:
    """Get current prompt content.

    Returns:
        PromptResponse with prompt name and content
    """
    prompt_name = settings_module.get_setting("prompt", "default")
    language = settings_module.get_setting("language", "en")
    content = settings_module.load_prompt_content(prompt_name, language=language)
    is_custom = prompt_name == "custom" and settings_module.custom_prompt_exists()

    return PromptResponse(
        prompt=prompt_name,
        content=content,
        is_custom=is_custom,
    )


@app.post("/prompt", response_model=PromptResponse)
async def save_prompt(req: PromptUpdateRequest) -> PromptResponse:
    """Save custom prompt content.

    Args:
        req: PromptUpdateRequest with content to save

    Returns:
        PromptResponse with saved prompt info
    """
    # Save to custom.md
    settings_module.save_custom_prompt(req.content)

    # Update settings to use custom prompt
    current = settings_module.load_settings()
    current["prompt"] = "custom"
    settings_module.save_settings(current)

    logger.info("Custom prompt saved and activated")

    return PromptResponse(
        prompt="custom",
        content=req.content,
        is_custom=True,
    )


@app.get("/greetings", response_model=GreetingsResponse)
async def get_greetings(language: str = "en") -> GreetingsResponse:
    """Get wake greetings for a language.

    Args:
        language: ISO 639-1 language code (default "en")

    Returns:
        GreetingsResponse with language and content (one greeting per line)
    """
    greetings = settings_module.load_greetings(language)
    return GreetingsResponse(
        language=language,
        content="\n".join(greetings),
    )


@app.post("/greetings", response_model=GreetingsResponse)
async def save_greetings(req: GreetingsUpdateRequest) -> GreetingsResponse:
    """Save wake greetings for a language.

    Args:
        req: GreetingsUpdateRequest with language and content

    Returns:
        GreetingsResponse with saved data
    """
    settings_module.save_greetings(req.language, req.content)
    logger.info(f"Greetings saved for language: {req.language}")
    return GreetingsResponse(
        language=req.language,
        content=req.content,
    )


# Default Piper voices (model IDs from HuggingFace speaches-ai)
# Piper bakes voice into model ID, so each "voice" is actually a model
PIPER_VOICES = [
    # English
    "speaches-ai/piper-en_US-ryan-high",
    "speaches-ai/piper-en_US-ljspeech-medium",
    "speaches-ai/piper-en_US-hfc_female-medium",
    "speaches-ai/piper-en_US-lessac-medium",
    "speaches-ai/piper-en_GB-aru-medium",
    "speaches-ai/piper-en_GB-alba-medium",
    # German
    "speaches-ai/piper-de_DE-eva_k-x_low",
    "speaches-ai/piper-de_DE-kerstin-low",
    "speaches-ai/piper-de_DE-thorsten-high",
    # French
    "speaches-ai/piper-fr_FR-mls-medium",
    "speaches-ai/piper-fr_FR-siwis-medium",
    # Spanish
    "speaches-ai/piper-es_ES-davefx-medium",
    "speaches-ai/piper-es_MX-ald-medium",
    # Russian
    "speaches-ai/piper-ru_RU-irina-medium",
    # Other
    "speaches-ai/piper-it_IT-riccardo-x_low",
    "speaches-ai/piper-pl_PL-darkman-medium",
    "speaches-ai/piper-pt_BR-faber-medium",
    "speaches-ai/piper-sk_SK-lili-medium",
    "speaches-ai/piper-uk_UA-lada-x_low",
]


@app.get("/voices", response_model=VoicesResponse)
async def get_voices(provider: str | None = None) -> VoicesResponse:
    """Get available TTS voices based on TTS provider.

    Args:
        provider: Optional provider override ("kokoro" or "piper").
                  If not specified, uses current setting.

    Returns:
        VoicesResponse with list of voice IDs (Kokoro) or model IDs (Piper)
    """
    if provider is None:
        settings = settings_module.load_settings()
        provider = settings.get("tts_provider", "kokoro")

    if provider == "piper":
        # Piper voices are model IDs - return curated list
        return VoicesResponse(voices=PIPER_VOICES)

    # Kokoro - fetch from API
    kokoro_url = os.getenv("KOKORO_URL", "http://kokoro:8880")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{kokoro_url}/v1/audio/voices",
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            # Kokoro returns {"voices": ["am_puck", ...]} as plain strings
            raw_voices = data.get("voices", [])
            if raw_voices and isinstance(raw_voices[0], str):
                # Plain string list
                voices = raw_voices
            else:
                # Object list with id field
                voices = [v.get("id") or v.get("voice_id") for v in raw_voices]
                voices = [v for v in voices if v]  # Filter None values

            return VoicesResponse(voices=voices)
    except Exception as e:
        logger.warning(f"Failed to fetch voices from Kokoro: {e}")
        # Return default voices as fallback
        return VoicesResponse(
            voices=["af_heart", "af_bella", "af_sarah", "am_adam", "am_puck"]
        )


class DownloadModelRequest(BaseModel):
    """Request body for /download-piper-model endpoint."""

    model_id: str


class DownloadModelResponse(BaseModel):
    """Response body for /download-piper-model endpoint."""

    success: bool
    message: str


@app.post("/download-piper-model", response_model=DownloadModelResponse)
async def download_piper_model(request: DownloadModelRequest) -> DownloadModelResponse:
    """Download a Piper TTS model to Speaches.

    Args:
        request: Request containing model_id (e.g., "speaches-ai/piper-en_US-ljspeech-medium")

    Returns:
        DownloadModelResponse with success status and message
    """
    speaches_url = os.getenv("SPEACHES_URL", "http://speaches:8000")
    model_id = request.model_id

    logger.info(f"Piper model download requested: {model_id}")

    # Validate it's a Piper model
    if not model_id.startswith("speaches-ai/piper-"):
        logger.warning(f"Invalid Piper model ID: {model_id}")
        return DownloadModelResponse(
            success=False,
            message=f"Invalid Piper model ID: {model_id}"
        )

    try:
        async with httpx.AsyncClient() as client:
            # Check if already downloaded
            check_response = await client.get(
                f"{speaches_url}/v1/models/{model_id}",
                timeout=5.0,
            )
            if check_response.status_code == 200:
                logger.info(f"Piper model already installed: {model_id}")
                return DownloadModelResponse(
                    success=True,
                    message=f"Model '{model_id}' already installed"
                )

            # Download the model (~60MB, should take <30s)
            logger.info(f"Downloading Piper model: {model_id}")
            response = await client.post(
                f"{speaches_url}/v1/models/{model_id}",
                timeout=60.0,
            )
            response.raise_for_status()

            logger.info(f"Piper model downloaded successfully: {model_id}")
            return DownloadModelResponse(
                success=True,
                message=f"Model '{model_id}' downloaded successfully"
            )
    except httpx.TimeoutException:
        logger.error(f"Timeout downloading Piper model: {model_id}")
        return DownloadModelResponse(
            success=False,
            message=f"Timeout downloading model '{model_id}'"
        )
    except Exception as e:
        logger.error(f"Failed to download Piper model {model_id}: {e}")
        return DownloadModelResponse(
            success=False,
            message=f"Failed to download model: {e}"
        )


@app.get("/models", response_model=ModelsResponse)
async def get_models() -> ModelsResponse:
    """Get available LLM models from Ollama.

    Returns:
        ModelsResponse with list of model names
    """
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ollama_host}/api/tags",
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            # Ollama returns {"models": [{"name": "...", ...}, ...]}
            models = [m.get("name") for m in data.get("models", [])]
            models = [m for m in models if m]  # Filter None values

            return ModelsResponse(models=models)
    except Exception as e:
        logger.warning(f"Failed to fetch models from Ollama: {e}")
        # Return empty list on failure
        return ModelsResponse(models=[])


# =============================================================================
# Wake Word Control Endpoints
# =============================================================================


class WakeWordStatusResponse(BaseModel):
    """Response body for /wake-word/status endpoint."""

    enabled: bool
    model: str
    threshold: float
    timeout: float


class WakeWordUpdateRequest(BaseModel):
    """Request body for wake word enable/disable."""

    enabled: bool


@app.get("/wake-word/status", response_model=WakeWordStatusResponse)
async def get_wake_word_status() -> WakeWordStatusResponse:
    """Get current wake word detection status.

    Returns:
        WakeWordStatusResponse with enabled state and configuration
    """
    settings = settings_module.load_settings()

    return WakeWordStatusResponse(
        enabled=settings.get("wake_word_enabled", False),
        model=settings.get("wake_word_model", "models/hey_jarvis.onnx"),
        threshold=settings.get("wake_word_threshold", 0.5),
        timeout=settings.get("wake_word_timeout", 3.0),
    )


@app.post("/wake-word/enable", response_model=WakeWordStatusResponse)
async def enable_wake_word() -> WakeWordStatusResponse:
    """Enable wake word detection.

    Note: This updates the setting but requires agent restart to take effect.

    Returns:
        WakeWordStatusResponse with updated configuration
    """
    current = settings_module.load_settings()
    current["wake_word_enabled"] = True
    settings_module.save_settings(current)

    settings = settings_module.reload_settings()
    logger.info("Wake word detection enabled (requires agent restart)")

    return WakeWordStatusResponse(
        enabled=True,
        model=settings.get("wake_word_model", "models/hey_jarvis.onnx"),
        threshold=settings.get("wake_word_threshold", 0.5),
        timeout=settings.get("wake_word_timeout", 3.0),
    )


@app.post("/wake-word/disable", response_model=WakeWordStatusResponse)
async def disable_wake_word() -> WakeWordStatusResponse:
    """Disable wake word detection.

    Note: This updates the setting but requires agent restart to take effect.

    Returns:
        WakeWordStatusResponse with updated configuration
    """
    current = settings_module.load_settings()
    current["wake_word_enabled"] = False
    settings_module.save_settings(current)

    settings = settings_module.reload_settings()
    logger.info("Wake word detection disabled (requires agent restart)")

    return WakeWordStatusResponse(
        enabled=False,
        model=settings.get("wake_word_model", "models/hey_jarvis.onnx"),
        threshold=settings.get("wake_word_threshold", 0.5),
        timeout=settings.get("wake_word_timeout", 3.0),
    )


class WakeWordModelsResponse(BaseModel):
    """Response containing available wake word models."""

    models: list[str]


@app.get("/wake-word/models", response_model=WakeWordModelsResponse)
async def get_wake_word_models() -> WakeWordModelsResponse:
    """List available wake word models.

    Scans the models/ directory for .onnx files, excluding infrastructure
    models (embedding_model, melspectrogram).

    Returns:
        WakeWordModelsResponse with list of model paths
    """
    from pathlib import Path

    models_dir = Path("models")
    models = []

    if models_dir.exists():
        for f in models_dir.glob("*.onnx"):
            # Skip infrastructure models used by OpenWakeWord
            if f.name not in ("embedding_model.onnx", "melspectrogram.onnx"):
                models.append(f"models/{f.name}")

    return WakeWordModelsResponse(models=sorted(models))


# =============================================================================
# Setup Wizard Endpoints
# =============================================================================


class SetupStatusResponse(BaseModel):
    """Response body for /setup/status endpoint."""

    completed: bool


class SetupCompleteRequest(BaseModel):
    """Request body for /setup/complete endpoint."""

    llm_provider: str  # "ollama" | "groq" | "openai_compatible" | "openrouter"
    # STT provider (independent from LLM)
    stt_provider: str | None = None  # "speaches" | "groq"
    # Ollama settings
    ollama_host: str | None = None
    ollama_model: str | None = None
    # Groq settings
    groq_api_key: str | None = None
    groq_model: str | None = None
    # OpenAI-compatible settings
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None
    # OpenRouter settings
    openrouter_api_key: str | None = None
    openrouter_model: str | None = None
    # TTS provider
    tts_provider: str = "kokoro"  # "kokoro" | "piper"
    tts_voice_kokoro: str | None = None
    tts_voice_piper: str | None = None
    # Language setting (optional, defaults to "en")
    language: str | None = None
    # Integrations (optional) - None means "don't change"
    hass_enabled: bool | None = None
    hass_host: str | None = None
    hass_token: str | None = None
    n8n_enabled: bool | None = None
    n8n_url: str | None = None
    n8n_token: str | None = None
    n8n_api_key: str | None = None


class SetupCompleteResponse(BaseModel):
    """Response body for /setup/complete endpoint."""

    success: bool
    message: str


class TestConnectionResponse(BaseModel):
    """Response body for connection test endpoints."""

    success: bool
    error: str | None = None
    models: list[str] | None = None  # For Ollama
    device_count: int | None = None  # For Home Assistant
    workflow_count: int | None = None  # For n8n


class TestOllamaRequest(BaseModel):
    """Request body for /setup/test-ollama endpoint."""

    host: str


class TestGroqRequest(BaseModel):
    """Request body for /setup/test-groq endpoint."""

    api_key: str


class TestHassRequest(BaseModel):
    """Request body for /setup/test-hass endpoint."""

    host: str
    token: str


class TestN8nRequest(BaseModel):
    """Request body for /setup/test-n8n endpoint."""

    url: str
    token: str | None = None


class TestOpenAICompatibleRequest(BaseModel):
    """Request body for /setup/test-openai-compatible endpoint."""

    base_url: str  # e.g., "http://localhost:8000/v1"
    api_key: str = ""  # Optional - some servers don't need auth


class TestOpenRouterRequest(BaseModel):
    """Request body for /setup/test-openrouter endpoint."""

    api_key: str  # Required - OpenRouter always needs API key


@app.get("/setup/status", response_model=SetupStatusResponse)
async def get_setup_status() -> SetupStatusResponse:
    """Check if first-launch setup has been completed.

    Returns:
        SetupStatusResponse with completed flag
    """
    settings = settings_module.load_settings()
    return SetupStatusResponse(completed=settings.get("first_launch_completed", False))


@app.post("/setup/complete", response_model=SetupCompleteResponse)
async def complete_setup(req: SetupCompleteRequest) -> SetupCompleteResponse:
    """Complete the first-launch setup wizard.

    Saves all settings and marks setup as completed.
    UI sets both stt_provider and llm_provider together based on llm_provider choice.

    Args:
        req: SetupCompleteRequest with provider and integration settings

    Returns:
        SetupCompleteResponse with success status
    """
    try:
        current = settings_module.load_settings()

        # Provider settings - STT is independent from LLM
        current["llm_provider"] = req.llm_provider
        current["stt_provider"] = req.stt_provider or "speaches"

        # LLM provider-specific settings
        if req.ollama_host:
            current["ollama_host"] = req.ollama_host
        if req.ollama_model:
            current["ollama_model"] = req.ollama_model
        if req.groq_model:
            current["groq_model"] = req.groq_model
        if req.openai_base_url:
            current["openai_base_url"] = req.openai_base_url
        if req.openai_api_key:
            current["openai_api_key"] = req.openai_api_key
        if req.openai_model:
            current["openai_model"] = req.openai_model
        if req.openrouter_api_key:
            current["openrouter_api_key"] = req.openrouter_api_key
        if req.openrouter_model:
            current["openrouter_model"] = req.openrouter_model

        # Home Assistant integration - only update if explicitly provided
        if req.hass_enabled is not None:
            current["hass_enabled"] = req.hass_enabled
            if req.hass_enabled:
                if req.hass_host:
                    current["hass_host"] = req.hass_host
                if req.hass_token:
                    current["hass_token"] = req.hass_token

        # n8n integration - only update if explicitly provided
        if req.n8n_enabled is not None:
            current["n8n_enabled"] = req.n8n_enabled
            if req.n8n_enabled:
                if req.n8n_url:
                    current["n8n_url"] = req.n8n_url
                if req.n8n_token:
                    current["n8n_token"] = req.n8n_token
                if req.n8n_api_key:
                    current["n8n_api_key"] = req.n8n_api_key

        # TTS provider and voice settings
        current["tts_provider"] = req.tts_provider
        if req.tts_voice_kokoro:
            current["tts_voice_kokoro"] = req.tts_voice_kokoro
        if req.tts_voice_piper:
            current["tts_voice_piper"] = req.tts_voice_piper

        # Language setting
        if req.language:
            current["language"] = req.language

        # Mark setup as complete
        current["first_launch_completed"] = True

        # Save Groq API key to settings (before save_settings call)
        if req.groq_api_key:
            current["groq_api_key"] = req.groq_api_key

        # Save settings
        settings_module.save_settings(current)

        logger.info("First-launch setup completed")
        return SetupCompleteResponse(success=True, message="Setup completed successfully")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return SetupCompleteResponse(success=False, message=str(e))


@app.post("/setup/test-ollama", response_model=TestConnectionResponse)
async def test_ollama(req: TestOllamaRequest) -> TestConnectionResponse:
    """Test Ollama connection and list available models.

    Args:
        req: TestOllamaRequest with host URL

    Returns:
        TestConnectionResponse with success status and model list
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{req.host}/api/tags",
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            models = [m.get("name") for m in data.get("models", [])]
            models = [m for m in models if m]

            return TestConnectionResponse(success=True, models=models)
    except httpx.ConnectError:
        return TestConnectionResponse(
            success=False, error=f"Cannot connect to Ollama at {req.host}"
        )
    except Exception as e:
        return TestConnectionResponse(success=False, error=str(e))


@app.post("/setup/test-groq", response_model=TestConnectionResponse)
async def test_groq(req: TestGroqRequest) -> TestConnectionResponse:
    """Test Groq API key validity.

    Args:
        req: TestGroqRequest with API key

    Returns:
        TestConnectionResponse with success status
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {req.api_key}"},
                timeout=10.0,
            )
            if response.status_code == 401:
                return TestConnectionResponse(success=False, error="Invalid API key")
            response.raise_for_status()

            data = response.json()
            models = [m.get("id") for m in data.get("data", [])]
            models = [m for m in models if m]

            return TestConnectionResponse(success=True, models=models)
    except Exception as e:
        return TestConnectionResponse(success=False, error=str(e))


@app.post("/setup/test-hass", response_model=TestConnectionResponse)
async def test_hass(req: TestHassRequest) -> TestConnectionResponse:
    """Test Home Assistant connection.

    Args:
        req: TestHassRequest with host and token

    Returns:
        TestConnectionResponse with success status and device count
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{req.host}/api/states",
                headers={"Authorization": f"Bearer {req.token}"},
                timeout=10.0,
            )
            if response.status_code == 401:
                return TestConnectionResponse(
                    success=False, error="Invalid access token"
                )
            response.raise_for_status()

            states = response.json()
            device_count = len(states)

            return TestConnectionResponse(success=True, device_count=device_count)
    except httpx.ConnectError:
        return TestConnectionResponse(
            success=False, error=f"Cannot connect to Home Assistant at {req.host}"
        )
    except Exception as e:
        return TestConnectionResponse(success=False, error=str(e))


@app.post("/setup/test-n8n", response_model=TestConnectionResponse)
async def test_n8n(req: TestN8nRequest) -> TestConnectionResponse:
    """Test n8n MCP connection.

    Args:
        req: TestN8nRequest with URL and token

    Returns:
        TestConnectionResponse with success status
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if req.token:
            headers["Authorization"] = f"Bearer {req.token}"

        async with httpx.AsyncClient() as client:
            # MCP protocol: list tools to verify connection
            response = await client.post(
                req.url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {},
                },
                timeout=10.0,
            )

            # Check for auth error in response body (n8n returns 200 with error message)
            if response.status_code == 200:
                text = response.text
                if "Unauthorized" in text:
                    return TestConnectionResponse(
                        success=False, error="Invalid access token"
                    )
                # SSE response with tools means success
                if "search_workflows" in text:
                    return TestConnectionResponse(success=True)

            if response.status_code == 401:
                return TestConnectionResponse(
                    success=False, error="Invalid access token"
                )
            response.raise_for_status()

            return TestConnectionResponse(success=True)
    except httpx.ConnectError:
        return TestConnectionResponse(
            success=False, error=f"Cannot connect to n8n at {req.url}"
        )
    except Exception as e:
        return TestConnectionResponse(success=False, error=str(e))


@app.post("/setup/test-openai-compatible", response_model=TestConnectionResponse)
async def test_openai_compatible(req: TestOpenAICompatibleRequest) -> TestConnectionResponse:
    """Test OpenAI-compatible server connection and list available models.

    Args:
        req: TestOpenAICompatibleRequest with base URL and optional API key

    Returns:
        TestConnectionResponse with success status and model list
    """
    # Normalize URL - strip trailing slash
    base_url = req.base_url.rstrip("/")

    try:
        headers = {}
        if req.api_key:
            headers["Authorization"] = f"Bearer {req.api_key}"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/models",
                headers=headers,
                timeout=10.0,
            )

            if response.status_code == 401:
                return TestConnectionResponse(success=False, error="Invalid API key")
            response.raise_for_status()

            data = response.json()

            # Handle both OpenAI format {"data": [...]} and alternative {"models": [...]}
            model_list = data.get("data") or data.get("models") or []

            # Extract model IDs - handle both dict and string formats
            models = []
            for m in model_list:
                if isinstance(m, dict):
                    model_id = m.get("id") or m.get("name")
                    if model_id:
                        models.append(model_id)
                elif isinstance(m, str):
                    models.append(m)

            return TestConnectionResponse(success=True, models=models)

    except httpx.ConnectError:
        return TestConnectionResponse(
            success=False,
            error=f"Cannot connect to server at {base_url}"
        )
    except httpx.TimeoutException:
        return TestConnectionResponse(
            success=False,
            error=f"Connection timed out - server at {base_url} may be slow or unreachable"
        )
    except Exception as e:
        return TestConnectionResponse(success=False, error=str(e))


@app.post("/setup/test-openrouter", response_model=TestConnectionResponse)
async def test_openrouter(req: TestOpenRouterRequest) -> TestConnectionResponse:
    """Test OpenRouter API key and list available models with tool support.

    Args:
        req: TestOpenRouterRequest with API key

    Returns:
        TestConnectionResponse with success status and tool-capable model list
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                params={"supported_parameters": "tools"},
                headers={"Authorization": f"Bearer {req.api_key}"},
                timeout=15.0,  # Longer timeout for large response
            )

            if response.status_code == 401:
                return TestConnectionResponse(success=False, error="Invalid API key")
            response.raise_for_status()

            data = response.json()
            models = [m.get("id") for m in data.get("data", [])]
            models = [m for m in models if m]

            return TestConnectionResponse(success=True, models=models)

    except httpx.TimeoutException:
        return TestConnectionResponse(
            success=False,
            error="Request timed out - try again"
        )
    except Exception as e:
        return TestConnectionResponse(success=False, error=str(e))


# =============================================================================
# n8n Workflows Endpoint
# =============================================================================


class N8nWorkflowItem(BaseModel):
    """Individual workflow in the n8n workflows list."""

    id: str
    name: str
    active: bool
    tags: list[str]
    createdAt: str  # noqa: N815 (matches n8n API JSON field)
    updatedAt: str  # noqa: N815 (matches n8n API JSON field)
    caal_registry_id: str | None = None
    caal_registry_version: str | None = None


class N8nWorkflowsResponse(BaseModel):
    """Response body for /n8n-workflows endpoint."""

    workflows: list[N8nWorkflowItem]
    n8n_base_url: str  # Base URL for building workflow links


class N8nWorkflowDetailResponse(BaseModel):
    """Response body for /n8n-workflow/{id} endpoint."""

    workflow: dict  # Full workflow JSON


@app.get("/n8n-workflows", response_model=N8nWorkflowsResponse)
async def get_n8n_workflows() -> N8nWorkflowsResponse:
    """Get all n8n workflows with CAAL registry tracking info.

    This endpoint:
    1. Fetches workflow list from n8n MCP server
    2. Uses local cache for registry info (fast)
    3. For uncached workflows, fetches full workflow to parse sticky note
    4. Prunes cache entries for deleted workflows

    Returns:
        N8nWorkflowsResponse with list of workflows (including registry info) and n8n base URL

    Raises:
        HTTPException: 400 if n8n not configured, 500 if fetch fails
    """
    settings = settings_module.load_settings()

    # Check if n8n is enabled and configured
    n8n_enabled = settings.get("n8n_enabled", False)
    n8n_mcp_url = settings.get("n8n_url", "")
    n8n_token = settings.get("n8n_token", "")
    n8n_api_key = settings.get("n8n_api_key", "")

    if not n8n_enabled or not n8n_mcp_url:
        raise HTTPException(
            status_code=400,
            detail="n8n not configured. Enable n8n in settings first.",
        )

    # Extract base n8n URL (strip /mcp-server/http suffix)
    # e.g., http://192.168.86.47:5678/mcp-server/http -> http://192.168.86.47:5678
    n8n_base_url = n8n_mcp_url.replace("/mcp-server/http", "").rstrip("/")

    try:
        # Prepare headers for MCP requests
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if n8n_token:
            headers["Authorization"] = f"Bearer {n8n_token}"

        # Prepare headers for REST API requests
        api_headers = {}
        if n8n_api_key:
            api_headers["X-N8N-API-KEY"] = n8n_api_key

        workflows = []
        workflow_list = []

        async with httpx.AsyncClient() as client:
            # Call search_workflows MCP tool
            search_response = await client.post(
                n8n_mcp_url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "search_workflows",
                        "arguments": {},
                    },
                },
                timeout=30.0,
            )
            search_response.raise_for_status()

            # Parse SSE response from n8n MCP server
            search_text = search_response.text

            # Parse SSE lines
            for line in search_text.split('\n'):
                if line.startswith('data: '):
                    try:
                        sse_data = json.loads(line[6:])  # Strip 'data: ' prefix
                        if "result" in sse_data and "content" in sse_data["result"]:
                            content = sse_data["result"]["content"]
                            if isinstance(content, list) and len(content) > 0:
                                # MCP returns content as array of TextContent
                                workflow_data = json.loads(content[0]["text"])
                                workflow_list = workflow_data.get("data", [])
                                break
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.warning(f"Failed to parse SSE line: {e}")
                        continue

            logger.info(f"Found {len(workflow_list)} n8n workflows")

            # Get active workflow IDs and prune deleted workflows from cache
            active_ids = {wf["id"] for wf in workflow_list}
            pruned = registry_cache.prune_deleted_workflows(active_ids)
            if pruned > 0:
                logger.info(f"Pruned {pruned} deleted workflows from cache")

            # Identify uncached workflows
            uncached_wf_ids = []
            for wf in workflow_list:
                cached = registry_cache.get_cached_entry(wf["id"])
                if cached is None:
                    uncached_wf_ids.append(wf["id"])

            logger.info(f"Uncached workflows to fetch: {len(uncached_wf_ids)}")

            # Fetch uncached workflows and parse sticky notes
            for wf_id in uncached_wf_ids:
                try:
                    workflow_url = f"{n8n_base_url}/api/v1/workflows/{wf_id}"
                    response = await client.get(
                        workflow_url,
                        headers=api_headers,
                        timeout=30.0,
                    )
                    if response.status_code == 200:
                        workflow_full = response.json()
                        nodes = workflow_full.get("nodes", [])
                        entry = registry_cache.parse_sticky_note_registry_info(nodes)
                        registry_cache.set_cached_entry(
                            wf_id,
                            entry["registry_id"],
                            entry["version"],
                        )
                    else:
                        # Couldn't fetch, mark as custom to avoid refetching
                        registry_cache.set_cached_entry(wf_id, None, None)
                except Exception as e:
                    logger.warning(f"Failed to fetch workflow {wf_id}: {e}")
                    # Mark as custom to avoid refetching
                    registry_cache.set_cached_entry(wf_id, None, None)

            # Build response items with registry info from cache
            for wf in workflow_list:
                cached = registry_cache.get_cached_entry(wf["id"])
                workflows.append(
                    N8nWorkflowItem(
                        id=wf["id"],
                        name=wf["name"],
                        active=wf.get("active", False),
                        tags=wf.get("tags", []),
                        createdAt=wf.get("createdAt", ""),
                        updatedAt=wf.get("updatedAt", ""),
                        caal_registry_id=cached["registry_id"] if cached else None,
                        caal_registry_version=cached["version"] if cached else None,
                    )
                )

        return N8nWorkflowsResponse(workflows=workflows, n8n_base_url=n8n_base_url)

    except httpx.ConnectError:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot connect to n8n at {n8n_base_url}",
        )
    except Exception as e:
        logger.error(f"Failed to fetch n8n workflows: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch workflows: {str(e)}",
        )


@app.get("/n8n-workflow/{workflow_id}", response_model=N8nWorkflowDetailResponse)
async def get_n8n_workflow(workflow_id: str) -> N8nWorkflowDetailResponse:
    """Get full workflow JSON for a single workflow.

    Fetches complete workflow data via n8n REST API, including credentials.
    Used for workflow submission to the Tool Registry.

    Args:
        workflow_id: The n8n workflow ID

    Returns:
        N8nWorkflowDetailResponse with full workflow JSON

    Raises:
        HTTPException: 400 if n8n not configured, 404 if not found, 500 if fetch fails
    """
    settings = settings_module.load_settings()

    # Check if n8n is enabled and configured
    n8n_enabled = settings.get("n8n_enabled", False)
    n8n_mcp_url = settings.get("n8n_url", "")
    n8n_api_key = settings.get("n8n_api_key", "")

    if not n8n_enabled or not n8n_mcp_url:
        raise HTTPException(
            status_code=400,
            detail="n8n not configured. Enable n8n in settings first.",
        )

    # Extract base n8n URL (strip /mcp-server/http suffix)
    n8n_base_url = n8n_mcp_url.replace("/mcp-server/http", "").rstrip("/")

    try:
        async with httpx.AsyncClient() as client:
            # Fetch workflow via n8n REST API
            api_headers = {}
            if n8n_api_key:
                api_headers["X-N8N-API-KEY"] = n8n_api_key

            workflow_url = f"{n8n_base_url}/api/v1/workflows/{workflow_id}"
            logger.debug(f"Fetching workflow from: {workflow_url}")

            response = await client.get(
                workflow_url,
                headers=api_headers,
                timeout=30.0,
            )

            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Workflow {workflow_id} not found",
                )

            response.raise_for_status()

            # Parse JSON response from REST API
            workflow_full = response.json()

            return N8nWorkflowDetailResponse(workflow=workflow_full)

    except HTTPException:
        raise
    except httpx.ConnectError:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot connect to n8n at {n8n_base_url}",
        )
    except Exception as e:
        logger.error(f"Failed to fetch workflow {workflow_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch workflow: {str(e)}",
        )


# =============================================================================
# Prewarm Endpoint
# =============================================================================


class PrewarmResponse(BaseModel):
    """Response body for /prewarm endpoint."""

    status: str
    message: str


@app.post("/prewarm", response_model=PrewarmResponse)
async def prewarm() -> PrewarmResponse:
    """Trigger model preloading for Ollama.

    This endpoint triggers the preload_models() function to warm up
    the STT and LLM models. It returns immediately - preloading happens
    in background.

    Returns:
        PrewarmResponse with status
    """
    settings = settings_module.load_settings()

    # Only prewarm if using Ollama
    if settings.get("llm_provider", "ollama") != "ollama":
        return PrewarmResponse(
            status="skipped",
            message="Prewarm only applies to Ollama provider",
        )

    # Import and call preload in background
    import asyncio

    async def do_prewarm():
        try:
            # Import here to avoid circular imports
            from voice_agent import preload_models

            ollama_host = settings.get("ollama_host", "http://localhost:11434")
            ollama_model = settings.get("ollama_model", "ministral-3:8b")
            await preload_models(ollama_host, ollama_model)
            logger.info("Model prewarm completed")
        except Exception as e:
            logger.error(f"Prewarm failed: {e}")

    # Fire and forget
    asyncio.create_task(do_prewarm())

    return PrewarmResponse(
        status="started",
        message="Model preloading started in background",
    )
