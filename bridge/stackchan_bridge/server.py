"""xiaozhi WebSocket server: one Session per connected StackChan."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import uuid
import wave
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Awaitable, Callable

import httpx
import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request

from . import services
from .audio import FRAME_MS, IN_RATE, OUT_RATE, OpusDecoder, Vad, encode_opus_frames, normalize, rms
from .config import Config
from .text import clean, is_meaningful, split_sentences

log = logging.getLogger("stackchan_bridge")

WS_PATH = "/xiaozhi/v1"
PREROLL_MS = 300  # kept before voice onset so the first syllable is not cut
SEND_LEAD_FRAMES = 5  # frames sent ahead of real time to absorb Wi-Fi jitter


@dataclass
class Backends:
    transcribe: Callable[[bytes], Awaitable[str]]
    ask: Callable[[str], Awaitable[str]]
    synthesize: Callable[[str], Awaitable[tuple[np.ndarray, int]]]


def daily_conversation(base: str, now: float | None = None) -> str:
    """One Hermes session per local calendar day, so history does not grow forever."""
    return f"{base}-{time.strftime('%Y-%m-%d', time.localtime(now))}"


def real_backends(cfg: Config, http: httpx.AsyncClient) -> Backends:
    return Backends(
        transcribe=lambda pcm: services.transcribe(cfg.stt_uri, pcm, cfg.stt_language, cfg.stt_timeout_s),
        ask=lambda text: services.ask_hermes(http, cfg.hermes_url, cfg.hermes_api_key,
                                             daily_conversation(cfg.hermes_conversation),
                                             cfg.instructions, text, cfg.hermes_timeout_s),
        synthesize=lambda text: services.synthesize(cfg.tts_uri, text, cfg.tts_timeout_s),
    )


class AckAudio:
    """The acknowledgement phrase, synthesized once and shared by all sessions."""

    def __init__(self, text: str, synthesize: Callable[[str], Awaitable[tuple[np.ndarray, int]]]) -> None:
        self.text, self.synthesize = text, synthesize
        self.audio: tuple[np.ndarray, int] | None = None

    async def get(self) -> tuple[np.ndarray, int] | None:
        if self.audio is None and self.text:
            try:
                samples, rate = await self.synthesize(self.text)
            except services.ServiceError as exc:
                log.error("ack TTS failed, replying without it: %s", exc)  # retried next turn
                return None
            self.audio = loud(samples), rate
        return self.audio


def loud(samples: np.ndarray) -> np.ndarray:
    # canto-vits peaks ~0.4 of full scale; bring each sentence to full scale so the M5 speaker is audible.
    return np.frombuffer(normalize(samples.tobytes(), target_peak=1.0, max_gain=3.0), np.int16)


class Session:
    def __init__(self, ws: ServerConnection, cfg: Config, backends: Backends, ack: AckAudio) -> None:
        self.ws = ws
        self.cfg = cfg
        self.backends = backends
        self.ack = ack
        self.session_id = uuid.uuid4().hex
        self.decoder = OpusDecoder()
        self.capturing = False
        self.turn: asyncio.Task | None = None
        self._reset_capture()

    def _reset_capture(self) -> None:
        self.vad = Vad(self.cfg.vad_rms_threshold, self.cfg.vad_start_speech_ms, self.cfg.vad_end_silence_ms)
        self.frames: list[bytes] = []
        self.captured_ms = 0  # audio received since listen start (not wall clock: frames can arrive in bursts)
        self.levels: list[float] = []  # per-frame RMS since listen start, for diagnostics

    async def send_json(self, **msg) -> None:
        await self.ws.send(json.dumps({"session_id": self.session_id, **msg}, ensure_ascii=False))

    async def run(self) -> None:
        try:
            async for message in self.ws:
                if isinstance(message, bytes):
                    await self.on_audio(message)
                else:
                    await self.on_json(json.loads(message))
        except ConnectionClosed:
            pass
        finally:
            if self.turn and not self.turn.done():
                self.turn.cancel()

    async def on_json(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "hello":
            await self.send_json(type="hello", transport="websocket", audio_params={
                "format": "opus", "sample_rate": OUT_RATE, "channels": 1, "frame_duration": FRAME_MS})
        elif kind == "listen":
            state = msg.get("state")
            if state == "start":
                self.capturing = True
                self._reset_capture()
            elif state == "stop" and self.capturing:
                await self.end_utterance("device stop")
            elif state == "detect":
                log.info("wake word: %s", msg.get("text", ""))
        elif kind == "abort":
            if self.turn and not self.turn.done():
                self.turn.cancel()
            await self.send_json(type="tts", state="stop")
        else:
            log.debug("ignored message type %r", kind)

    async def on_audio(self, packet: bytes) -> None:
        if not self.capturing:
            return
        pcm = self.decoder.decode(packet)
        self.captured_ms += FRAME_MS
        self.levels.append(rms(pcm))
        event = self.vad.feed(pcm)
        self.frames.append(pcm)
        if not self.vad.in_speech and event != "end":
            # VAD fires only after vad_start_speech_ms of voice, so keep that much plus the pre-roll.
            self.frames = self.frames[-((PREROLL_MS + self.cfg.vad_start_speech_ms) // FRAME_MS):]
            if self.captured_ms > self.cfg.listen_timeout_ms:
                log.info("no speech for %d ms, closing session", self.cfg.listen_timeout_ms)
                self.capturing = False
                # Close in the background: the loop must keep draining in-flight frames,
                # otherwise the device's close reply sits unread until close_timeout.
                asyncio.create_task(self.ws.close())
            return
        if event == "end" or len(self.frames) * FRAME_MS >= self.cfg.max_utterance_ms:
            await self.end_utterance("silence" if event == "end" else "max length")

    async def end_utterance(self, reason: str = "device stop") -> None:
        self.capturing = False
        voiced_ms = sum(FRAME_MS for level in self.levels if level >= self.cfg.vad_rms_threshold)
        levels = sorted(self.levels) or [0.0]
        log.info("utterance %d ms (ended by %s), rms p10 %.3f p50 %.3f p90 %.3f, threshold %.3f",
                 len(self.frames) * FRAME_MS, reason, levels[len(levels) // 10], levels[len(levels) // 2],
                 levels[len(levels) * 9 // 10], self.cfg.vad_rms_threshold)
        if voiced_ms < self.cfg.min_voiced_ms:
            log.info("discarded as noise: only %d ms voiced", voiced_ms)
            self.frames = []
            self.capturing = True  # device is still listening
            self._reset_capture()
            return
        pcm = normalize(b"".join(self.frames))
        self.frames = []
        if self.cfg.debug_audio_dir:
            save_wav(self.cfg.debug_audio_dir, pcm)
        self.turn = asyncio.create_task(self.run_turn(pcm))

    async def run_turn(self, pcm: bytes) -> None:
        t0 = time.monotonic()
        ms = lambda: int((time.monotonic() - t0) * 1000)  # noqa: E731
        try:
            transcript = await self.backends.transcribe(pcm)
        except services.ServiceError as exc:
            log.error("turn failed at STT: %s", exc)
            await self.send_json(type="alert", status="error", message="STT error", emotion="sad")
            await self.ws.close()
            return
        log.info("stt %d ms: %s", ms(), transcript)
        if not is_meaningful(transcript):
            self.capturing = True  # device is still listening; keep capturing
            self._reset_capture()
            return

        await self.send_json(type="stt", text=transcript)
        # Ask Hermes now; the acknowledgement plays while it works.
        reply = asyncio.create_task(self.ask(transcript))
        try:
            await self.speak(reply, t0)
        finally:
            reply.cancel()

    async def ask(self, transcript: str) -> str:
        t0 = time.monotonic()
        try:
            reply = await self.backends.ask(transcript)
            log.info("hermes %d ms: %s", int((time.monotonic() - t0) * 1000), reply)
        except services.ServiceError as exc:
            log.error("turn failed at Hermes: %s", exc)
            reply = self.cfg.fallback_text
        return clean(reply)

    async def speak(self, reply: Awaitable[str], t0: float) -> None:
        """One TTS session around ack + reply, so the device stays in Speaking (mic off) while Hermes works."""
        await self.send_json(type="llm", emotion="neutral")
        await self.send_json(type="tts", state="start")
        try:
            ack = await self.ack.get()
            if ack:
                log.info("ack audio %d ms after end of speech", int((time.monotonic() - t0) * 1000))
                await self.stream(_once(ack))
            await self.send_json(type="llm", emotion="working")
            await self.stream(self.reply_sentences(await reply, t0))
        except services.ServiceError as exc:
            log.error("turn failed at TTS: %s", exc)
            await self.send_json(type="alert", status="error", message="TTS error", emotion="sad")
        finally:
            await self.send_json(type="tts", state="stop")

    async def reply_sentences(self, text: str, t0: float):
        for i, sentence in enumerate(split_sentences(text)):
            samples, rate = await self.backends.synthesize(sentence)
            if i == 0:
                log.info("first reply audio %d ms after end of speech", int((time.monotonic() - t0) * 1000))
            await self.send_json(type="tts", state="sentence_start", text=sentence)
            yield loud(samples), rate

    async def stream(self, sentences) -> None:
        """Send Opus frames paced to real time (plus a small lead), continuously across sentences."""
        clock, sent = time.monotonic(), 0
        async for samples, rate in sentences:
            if sent == 0:
                clock = time.monotonic()
            for frame in encode_opus_frames(samples, rate):
                await self.ws.send(frame)
                sent += 1
                delay = clock + (sent - SEND_LEAD_FRAMES) * FRAME_MS / 1000 - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)


async def _once(item):
    yield item


def save_wav(directory: str, pcm: bytes) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    name = path / time.strftime("utt-%Y%m%dT%H%M%SZ.wav", time.gmtime())
    with wave.open(str(name), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(IN_RATE)
        w.writeframes(pcm)


def make_auth_check(cfg: Config):
    expected = f"Bearer {cfg.device_token}".encode()

    def process_request(connection: ServerConnection, request: Request):
        if not request.path.rstrip("/").startswith(WS_PATH):
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        given = request.headers.get("Authorization", "").encode()
        if not hmac.compare_digest(given, expected):
            log.warning("rejected connection from %s: bad token", connection.remote_address[0])
            return connection.respond(HTTPStatus.UNAUTHORIZED, "unauthorized\n")
        return None

    return process_request


async def serve_forever(cfg: Config, backends: Backends | None = None, ready: asyncio.Event | None = None) -> None:
    async with httpx.AsyncClient() as http:
        backends = backends or real_backends(cfg, http)
        ack = AckAudio(cfg.ack_text, backends.synthesize)

        async def handler(ws: ServerConnection) -> None:
            log.info("device connected from %s", ws.remote_address[0])
            await Session(ws, cfg, backends, ack).run()
            log.info("device disconnected")

        async with serve(handler, cfg.host, cfg.port, process_request=make_auth_check(cfg),
                         max_size=2**20, ping_interval=20,
                         # The device never answers the close frame; the 10 s default kept it "listening".
                         close_timeout=1) as server:
            log.info("listening on %s:%d%s/", cfg.host, cfg.port, WS_PATH)
            if ready:
                ready.set()
            await server.serve_forever()
