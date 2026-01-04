"""Wake word gated STT wrapper using OpenWakeWord.

Uses StreamAdapter for inner STT (since Speaches doesn't support WebSocket streaming).
Wake word detection gates when audio gets forwarded to the StreamAdapter.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from livekit import rtc
from livekit.agents.stt import (
    RecognizeStream,
    SpeechEvent,
    SpeechEventType,
    STT,
    STTCapabilities,
    StreamAdapter,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import aio
from livekit.agents.vad import VADEvent, VADEventType
from livekit.plugins import silero
from openwakeword.model import Model as OWWModel

logger = logging.getLogger(__name__)


class WakeWordState(str, Enum):
    """State of wake word detection."""

    LISTENING = "listening"
    ACTIVE = "active"


@dataclass
class WakeWordEvent:
    """Event emitted when wake word state changes."""

    state: WakeWordState
    model_name: str | None = None
    score: float | None = None


class WakeWordGatedSTT(STT):
    """STT wrapper that gates audio through OpenWakeWord detection.

    Audio is only forwarded to the inner STT when the wake word is detected.
    After a configurable silence timeout, returns to wake word listening mode.
    """

    def __init__(
        self,
        *,
        inner_stt: STT,
        model_path: str,
        threshold: float = 0.5,
        silence_timeout: float = 3.0,
        on_wake_detected: Callable[[], Awaitable[None]] | None = None,
        on_state_changed: Callable[[WakeWordState], Awaitable[None]] | None = None,
    ) -> None:
        """Initialize the wake word gated STT.

        Args:
            inner_stt: The actual STT to forward audio to after wake word detection.
            model_path: Path to the OpenWakeWord .onnx model file.
            threshold: Wake word detection threshold (0-1). Higher = more strict.
            silence_timeout: Seconds of silence before returning to listening mode.
            on_wake_detected: Callback when wake word is detected (e.g., trigger greeting).
            on_state_changed: Callback when state changes (for publishing to clients).
        """
        # Override capabilities to indicate we support streaming
        # Even though the inner STT may not support streaming, WE provide streaming
        # by gating audio through wake word detection before forwarding to inner STT
        super().__init__(
            capabilities=STTCapabilities(streaming=True, interim_results=False)
        )
        self._inner = inner_stt
        self._model_path = model_path
        self._threshold = threshold
        self._silence_timeout = silence_timeout
        self._on_wake_detected = on_wake_detected
        self._on_state_changed = on_state_changed
        self._oww: OWWModel | None = None
        self._active_stream: WakeWordGatedStream | None = None

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def provider(self) -> str:
        return self._inner.provider

    def _ensure_model(self) -> OWWModel:
        """Lazy-load the OpenWakeWord model."""
        if self._oww is None:
            try:
                logger.info(f"Loading OpenWakeWord model from {self._model_path}")
                self._oww = OWWModel(
                    wakeword_models=[self._model_path],
                    inference_framework="onnx",  # Use ONNX for .onnx models
                )
                logger.info("OpenWakeWord model loaded")
            except Exception as e:
                logger.error(f"Failed to load OpenWakeWord model from {self._model_path}: {e}")
                raise RuntimeError(f"Wake word model unavailable: {e}") from e
        return self._oww

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        # For non-streaming recognition, just pass through
        # Wake word gating only makes sense for streaming
        return await self._inner.recognize(
            buffer, language=language, conn_options=conn_options
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        stream = WakeWordGatedStream(
            stt=self,
            inner_stt=self._inner,
            oww=self._ensure_model(),
            threshold=self._threshold,
            silence_timeout=self._silence_timeout,
            on_wake_detected=self._on_wake_detected,
            on_state_changed=self._on_state_changed,
            language=language,
            conn_options=conn_options,
        )
        self._active_stream = stream
        return stream

    def set_agent_busy(self, busy: bool) -> None:
        """Set agent busy state - pauses silence timer while busy, resets when done."""
        if self._active_stream:
            self._active_stream.set_agent_busy(busy)

    async def aclose(self) -> None:
        await self._inner.aclose()


class WakeWordGatedStream(RecognizeStream):
    """Streaming STT that gates audio through wake word detection.

    Uses StreamAdapter for inner STT since Speaches doesn't support WebSocket streaming.
    Flow:
    1. LISTENING: Run wake word detection on all audio, discard if no trigger
    2. Wake word detected: Switch to ACTIVE, trigger greeting callback
    3. ACTIVE: Forward audio to StreamAdapter (which handles VAD + batch STT)
    4. Silence timeout: Return to LISTENING
    """

    # OpenWakeWord expects 16kHz mono audio, 80ms chunks (1280 samples)
    OWW_SAMPLE_RATE = 16000
    OWW_CHUNK_SAMPLES = 1280  # 80ms at 16kHz

    def __init__(
        self,
        stt: WakeWordGatedSTT,
        *,
        inner_stt: STT,
        oww: OWWModel,
        threshold: float,
        silence_timeout: float,
        on_wake_detected: Callable[[], Awaitable[None]] | None,
        on_state_changed: Callable[[WakeWordState], Awaitable[None]] | None,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        # Request 16kHz resampling for OpenWakeWord
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=self.OWW_SAMPLE_RATE,
        )
        self._inner_stt = inner_stt
        self._oww = oww
        self._threshold = threshold
        self._silence_timeout = silence_timeout
        self._on_wake_detected = on_wake_detected
        self._on_state_changed = on_state_changed
        self._language = language
        self._conn_options = conn_options

        self._state = WakeWordState.LISTENING
        self._oww_buffer: list[np.ndarray] = []  # Buffer for wake word detection
        self._last_speech_time: float = 0.0
        self._inner_stream: RecognizeStream | None = None
        self._agent_busy: bool = False  # True while agent is thinking/speaking
        self._speech_active: bool = False  # True while VAD detects speech

    def set_agent_busy(self, busy: bool) -> None:
        """Set agent busy state - pauses silence timer while busy, resets when done."""
        was_busy = self._agent_busy
        self._agent_busy = busy

        # When agent finishes (busy -> not busy), start fresh follow-up window
        if was_busy and not busy:
            self._last_speech_time = time.time()
            logger.info("Agent done, follow-up window started")

    async def _set_state(self, state: WakeWordState) -> None:
        """Update state and notify callback."""
        if self._state != state:
            self._state = state
            logger.info(f"Wake word state changed to: {state.value}")
            if self._on_state_changed:
                try:
                    await self._on_state_changed(state)
                except Exception as e:
                    logger.warning(f"Error in on_state_changed callback: {e}")

    async def _run(self) -> None:
        """Main processing loop."""

        # Create StreamAdapter to wrap the non-streaming STT with VAD
        vad = silero.VAD.load()
        stream_adapter = StreamAdapter(stt=self._inner_stt, vad=vad)

        # Create a separate VAD stream for tracking speech activity
        # This lets us know when the user is still speaking (not just when STT returns)
        speech_vad = silero.VAD.load()
        vad_stream = speech_vad.stream()

        # Get a stream from the adapter
        self._inner_stream = stream_adapter.stream(
            language=self._language,
            conn_options=self._conn_options,
        )

        # Set initial state
        await self._set_state(WakeWordState.LISTENING)

        async def _process_audio() -> None:
            """Process incoming audio frames."""
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    if self._state == WakeWordState.ACTIVE:
                        self._inner_stream.flush()
                        vad_stream.flush()
                    continue

                frame: rtc.AudioFrame = data

                if self._state == WakeWordState.LISTENING:
                    await self._process_wake_word(frame)
                else:
                    # Active mode - forward to StreamAdapter AND VAD tracker
                    self._inner_stream.push_frame(frame)
                    vad_stream.push_frame(frame)

            # End input when done
            self._inner_stream.end_input()
            vad_stream.end_input()

        async def _read_inner_events() -> None:
            """Read events from inner StreamAdapter and forward them."""
            async for event in self._inner_stream:
                self._event_ch.send_nowait(event)
                # Reset silence timer on any speech event (STT activity)
                if event.type in (
                    SpeechEventType.START_OF_SPEECH,
                    SpeechEventType.INTERIM_TRANSCRIPT,
                    SpeechEventType.FINAL_TRANSCRIPT,
                ):
                    self._last_speech_time = time.time()

        async def _track_speech_activity() -> None:
            """Track VAD events to know when user is speaking."""
            async for event in vad_stream:
                if event.type == VADEventType.START_OF_SPEECH:
                    self._speech_active = True
                    self._last_speech_time = time.time()
                    logger.debug("VAD: speech started")
                elif event.type == VADEventType.END_OF_SPEECH:
                    self._speech_active = False
                    self._last_speech_time = time.time()
                    logger.debug("VAD: speech ended")

        async def _monitor_silence() -> None:
            """Monitor for silence timeout to return to listening mode."""
            while True:
                await asyncio.sleep(0.5)  # Check every 500ms

                # Only timeout if active, agent is not busy, AND user is not speaking
                if (
                    self._state == WakeWordState.ACTIVE
                    and not self._agent_busy
                    and not self._speech_active
                ):
                    elapsed = time.time() - self._last_speech_time
                    if elapsed >= self._silence_timeout:
                        logger.info(
                            f"Silence timeout ({self._silence_timeout}s), "
                            "returning to wake word listening"
                        )
                        await self._set_state(WakeWordState.LISTENING)
                        # Reset OpenWakeWord model state for fresh detection
                        self._oww.reset()

        tasks = [
            asyncio.create_task(_process_audio(), name="process_audio"),
            asyncio.create_task(_read_inner_events(), name="read_inner_events"),
            asyncio.create_task(_track_speech_activity(), name="track_speech_activity"),
            asyncio.create_task(_monitor_silence(), name="monitor_silence"),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await aio.cancel_and_wait(*tasks)
            if self._inner_stream:
                await self._inner_stream.aclose()
            await vad_stream.aclose()

    async def _process_wake_word(self, frame: rtc.AudioFrame) -> None:
        """Process audio frame for wake word detection."""
        # Convert frame to numpy array (int16)
        audio_data = np.frombuffer(frame.data, dtype=np.int16)

        # Handle multi-channel by taking first channel
        if frame.num_channels > 1:
            audio_data = audio_data[::frame.num_channels]

        # Accumulate audio until we have enough for OpenWakeWord
        self._oww_buffer.append(audio_data)
        total_samples = sum(len(chunk) for chunk in self._oww_buffer)

        # Process when we have enough samples (80ms chunks)
        while total_samples >= self.OWW_CHUNK_SAMPLES:
            # Concatenate and take exactly what we need
            combined = np.concatenate(self._oww_buffer)
            chunk = combined[: self.OWW_CHUNK_SAMPLES]

            # Keep remainder for next iteration
            remainder = combined[self.OWW_CHUNK_SAMPLES :]
            self._oww_buffer = [remainder] if len(remainder) > 0 else []
            total_samples = len(remainder)

            # Run wake word detection
            predictions = self._oww.predict(chunk)

            for model_name, score in predictions.items():
                if score >= self._threshold:
                    detect_time = time.time()
                    logger.info(
                        f"Wake word detected! model={model_name}, score={score:.3f}"
                    )

                    # Trigger wake callback FIRST (e.g., greeting) - fire and forget
                    # Do this before state change to minimize latency
                    if self._on_wake_detected:
                        asyncio.create_task(self._on_wake_detected())
                        # Yield to event loop so the task can start immediately
                        await asyncio.sleep(0)

                    # Switch to active mode
                    await self._set_state(WakeWordState.ACTIVE)
                    self._last_speech_time = detect_time

                    return
