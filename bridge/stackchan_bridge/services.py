"""Clients for Wyoming STT/TTS, the Hermes api_server (legacy), and the OpenBot webhook brain."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid

import httpx
import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.tts import Synthesize

from .audio import IN_RATE


class ServiceError(Exception):
    """A backend (STT, TTS, Hermes, OpenBot webhook) failed; message is safe to log."""


async def transcribe(uri: str, pcm: bytes, language: str, timeout_s: float) -> str:
    async def run() -> str:
        async with AsyncClient.from_uri(uri) as client:
            await client.write_event(Transcribe(language=language).event())
            await client.write_event(AudioStart(rate=IN_RATE, width=2, channels=1).event())
            step = IN_RATE * 2  # 1 s chunks
            for i in range(0, len(pcm), step):
                await client.write_event(AudioChunk(rate=IN_RATE, width=2, channels=1, audio=pcm[i:i + step]).event())
            await client.write_event(AudioStop().event())
            while True:
                event = await client.read_event()
                if event is None:
                    raise ServiceError("STT closed connection without transcript")
                if Transcript.is_type(event.type):
                    return Transcript.from_event(event).text.strip()

    try:
        return await asyncio.wait_for(run(), timeout_s)
    except (OSError, asyncio.TimeoutError) as exc:
        raise ServiceError(f"STT unavailable: {type(exc).__name__}") from exc


async def synthesize(uri: str, text: str, timeout_s: float) -> tuple[np.ndarray, int]:
    async def run() -> tuple[np.ndarray, int]:
        rate, chunks = 0, []
        async with AsyncClient.from_uri(uri) as client:
            await client.write_event(Synthesize(text=text).event())
            while True:
                event = await client.read_event()
                if event is None:
                    raise ServiceError("TTS closed connection early")
                if AudioStart.is_type(event.type):
                    start = AudioStart.from_event(event)
                    if start.width != 2 or start.channels != 1:
                        raise ServiceError("TTS returned non 16-bit mono audio")
                    rate = start.rate
                elif AudioChunk.is_type(event.type):
                    chunk = AudioChunk.from_event(event)
                    rate = rate or chunk.rate
                    chunks.append(chunk.audio)
                elif AudioStop.is_type(event.type):
                    return np.frombuffer(b"".join(chunks), np.int16), rate

    try:
        return await asyncio.wait_for(run(), timeout_s)
    except (OSError, asyncio.TimeoutError) as exc:
        raise ServiceError(f"TTS unavailable: {type(exc).__name__}") from exc


async def ask_hermes(client: httpx.AsyncClient, url: str, api_key: str, conversation: str,
                     instructions: str, text: str, timeout_s: float, model: str = "", provider: str = "",
                     reasoning_effort: str = "") -> str:
    body = {"model": model or "hermes-agent", "conversation": conversation, "instructions": instructions, "input": text}
    if provider:  # Hermes ignores a bare model without a provider unless direct_model_requests is on
        body["provider"] = provider
    if reasoning_effort:
        body["model_options"] = {"reasoning_effort": reasoning_effort}
    try:
        response = await client.post(
            f"{url}/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=timeout_s,
        )
    except httpx.HTTPError as exc:
        raise ServiceError(f"Hermes unavailable: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise ServiceError(f"Hermes HTTP {response.status_code}")
    body = response.json()
    if body.get("error") or body.get("status") not in (None, "completed"):
        raise ServiceError(f"Hermes response status {body.get('status')!r}")
    reply = "".join(
        part.get("text", "")
        for item in body.get("output", []) if item.get("type") == "message"
        for part in item.get("content", []) if part.get("type") in ("output_text", "text")
    ).strip()
    if not reply:
        raise ServiceError("Hermes returned empty reply")
    return reply


# Spoken on the device as soon as the transcript reaches OpenBot; the full
# answer arrives later via the routine calling /announce.
WEBHOOK_ACK = "收到，管家跟緊。"

# A follow-up (device re-listening right after a reply) may just be the tail
# of the turn the routine already answered; still deliver it, flagged.
async def deliver_to_openbot(client: httpx.AsyncClient, url: str, secret: str, text: str,
                             followup: bool, timeout_s: float) -> str:
    """POST a voice transcript to the OpenBot webhook brain. Returns short text
    for the device to speak at once; the routine announces the real answer."""
    payload = {"text": text, "followup": followup, "device": "stackchan"}
    raw = json.dumps({"type": "stackchan_voice", "data": payload}, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    timestamp = str(int(time.time()))
    delivery_id = uuid.uuid4().hex
    signature = hmac.new(secret.encode("utf-8"),
                         f"{timestamp}.{delivery_id}.".encode("utf-8") + raw,
                         hashlib.sha256).hexdigest()
    try:
        response = await client.post(
            url,
            content=raw,
            headers={"Content-Type": "application/json",
                     "x-openbot-timestamp": timestamp,
                     "x-openbot-delivery-id": delivery_id,
                     "x-openbot-signature": f"sha256={signature}"},
            timeout=timeout_s,
        )
    except httpx.HTTPError as exc:
        raise ServiceError(f"OpenBot webhook unavailable: {type(exc).__name__}") from exc
    if response.status_code not in (200, 202):
        raise ServiceError(f"OpenBot webhook HTTP {response.status_code}")
    return WEBHOOK_ACK
