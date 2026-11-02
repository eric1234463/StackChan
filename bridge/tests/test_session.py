"""End-to-end over a real WebSocket with fake STT/Hermes/TTS backends."""
import asyncio
import json

import httpx
import numpy as np
import pytest
import websockets

from stackchan_bridge import server, services
from stackchan_bridge.audio import FRAME_MS, IN_RATE, OUT_RATE, OpusDecoder, opuslib
from stackchan_bridge.config import Config
from stackchan_bridge.server import Backends, serve_forever

TOKEN = "test-device-token"
ACK = Config(device_token="", hermes_api_key="").ack_text
FRAME = IN_RATE * FRAME_MS // 1000


def opus_frames(amplitude: float, count: int) -> list[bytes]:
    enc = opuslib.Encoder(IN_RATE, 1, opuslib.APPLICATION_VOIP)
    t = np.arange(FRAME) / IN_RATE
    pcm = (np.sin(2 * np.pi * 300 * t) * amplitude * 32767).astype(np.int16).tobytes()
    return [enc.encode(pcm, FRAME) for _ in range(count)]


class Fakes:
    def __init__(self, transcript="開客廳燈", reply="好，開咗。", hermes_error=False, hermes_gate=None):
        self.transcript, self.reply, self.hermes_error = transcript, reply, hermes_error
        self.hermes_gate = hermes_gate  # if set, Hermes answers only after the test opens it
        self.asked, self.spoken, self.stt_bytes = [], [], 0

    async def transcribe(self, pcm: bytes) -> str:
        self.stt_bytes = len(pcm)
        return self.transcript

    async def ask(self, text: str) -> str:
        self.asked.append(text)
        if self.hermes_gate:
            await self.hermes_gate.wait()
        if self.hermes_error:
            raise services.ServiceError("Hermes HTTP 500")
        return self.reply

    async def synthesize(self, text: str):
        self.spoken.append(text)
        return np.zeros(22050 // 5, np.int16), 22050  # 200 ms of silence at a non-24k rate

    def backends(self) -> Backends:
        return Backends(self.transcribe, self.ask, self.synthesize)


@pytest.fixture
async def bridge(unused_tcp_port):
    started = []

    async def start(fakes: Fakes):
        cfg = Config(device_token=TOKEN, hermes_api_key="unused", host="127.0.0.1", port=unused_tcp_port,
                     listen_timeout_ms=3000, followup_timeout_ms=1000)
        ready = asyncio.Event()
        task = asyncio.create_task(serve_forever(cfg, fakes.backends(), ready))
        started.append(task)
        await ready.wait()
        return f"ws://127.0.0.1:{unused_tcp_port}/xiaozhi/v1/"

    yield start
    for task in started:
        task.cancel()


async def handshake(ws):
    await ws.send(json.dumps({"type": "hello", "version": 1, "transport": "websocket",
                              "audio_params": {"format": "opus", "sample_rate": 16000, "channels": 1,
                                               "frame_duration": 60}}))
    hello = json.loads(await ws.recv())
    assert hello["type"] == "hello" and hello["transport"] == "websocket"
    assert hello["audio_params"]["sample_rate"] == OUT_RATE
    return hello["session_id"]


async def talk(ws):
    await ws.send(json.dumps({"type": "listen", "state": "start", "mode": "auto"}))
    for frame in opus_frames(0.0, 3) + opus_frames(0.3, 10) + opus_frames(0.0, 25):  # 1.5 s > 1200 ms end silence
        await ws.send(frame)


async def collect_turn(ws):
    texts, audio = [], []
    while True:
        msg = await asyncio.wait_for(ws.recv(), 5)
        if isinstance(msg, bytes):
            audio.append(msg)
            continue
        msg = json.loads(msg)
        texts.append(msg)
        if msg["type"] == "tts" and msg["state"] == "stop":
            return texts, audio


async def test_full_turn(bridge):
    fakes = Fakes()
    url = await bridge(fakes)
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
        session_id = await handshake(ws)
        await talk(ws)
        texts, audio = await collect_turn(ws)

    kinds = [(m["type"], m.get("state")) for m in texts]
    assert kinds == [("stt", None), ("llm", None), ("tts", "start"), ("llm", None), ("llm", None),
                     ("tts", "sentence_start"), ("tts", "stop")]
    assert [texts[3]["emotion"], texts[4]["emotion"]] == ["nod", "working"] and texts[5]["text"] == "好，開咗。"
    assert texts[0]["text"] == "開客廳燈" and all(m["session_id"] == session_id for m in texts)
    assert fakes.asked == ["開客廳燈"] and fakes.spoken == [ACK, "好，開咗。"]
    assert fakes.stt_bytes >= 10 * FRAME * 2  # speech frames reached STT (plus preroll)
    dec = OpusDecoder(OUT_RATE)
    assert sum(len(dec.decode(p)) for p in audio) >= OUT_RATE // 5 * 2  # >= 200 ms decodes at 24 kHz


async def test_hermes_failure_speaks_fallback_and_stops(bridge):
    fakes = Fakes(hermes_error=True)
    url = await bridge(fakes)
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
        await handshake(ws)
        await talk(ws)
        texts, _ = await collect_turn(ws)
    assert fakes.spoken == [ACK, "我而家連唔到 Hermes。"]
    assert texts[-1] == {"session_id": texts[-1]["session_id"], "type": "tts", "state": "stop"}


async def test_ack_plays_while_hermes_is_still_working(bridge):
    gate = asyncio.Event()
    fakes = Fakes(hermes_gate=gate)
    url = await bridge(fakes)
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
        await handshake(ws)
        await talk(ws)
        ack_frames = 0
        while True:  # Hermes is blocked: ack audio and the working state must still arrive
            msg = await asyncio.wait_for(ws.recv(), 5)
            if isinstance(msg, bytes):
                ack_frames += 1
            elif json.loads(msg).get("emotion") == "working":
                break
        assert fakes.asked == ["開客廳燈"] and ack_frames > 0
        gate.set()
        texts, _ = await collect_turn(ws)
    assert [m["text"] for m in texts if m.get("state") == "sentence_start"] == ["好，開咗。"]


async def test_ack_is_synthesized_once(bridge):
    fakes = Fakes()
    url = await bridge(fakes)
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
        await handshake(ws)
        for _ in range(2):
            await talk(ws)
            await collect_turn(ws)
    assert fakes.spoken == [ACK, "好，開咗。", "好，開咗。"]


async def test_filler_transcript_skips_hermes(bridge):
    fakes = Fakes(transcript="嗯")
    url = await bridge(fakes)
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
        await handshake(ws)
        await talk(ws)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ws.recv(), 1)
    assert fakes.asked == []


async def test_bad_token_rejected(bridge):
    url = await bridge(Fakes())
    with pytest.raises(websockets.InvalidStatus) as err:
        async with websockets.connect(url, additional_headers={"Authorization": "Bearer wrong"}):
            pass
    assert err.value.response.status_code == 401


async def test_silence_sends_device_idle_and_keeps_connection(bridge):
    fakes = Fakes()
    url = await bridge(fakes)
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
        await handshake(ws)
        await ws.send(json.dumps({"type": "listen", "state": "start", "mode": "auto"}))
        for frame in opus_frames(0.0, 60):  # 3.6 s of silence > listen_timeout_ms 3000
            await ws.send(frame)
        msg = json.loads(await asyncio.wait_for(ws.recv(), 5))
        assert {k: msg[k] for k in ("type", "command")} == {"type": "system", "command": "idle"}
        await talk(ws)  # still connected: the next wake word reuses this session
        await collect_turn(ws)
    assert fakes.asked == ["開客廳燈"]


async def test_follow_up_window_is_shorter_than_wake_window(bridge):
    url = await bridge(Fakes())  # fixture: listen 3000 ms, follow-up 1000 ms
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
        await handshake(ws)
        await talk(ws)
        await collect_turn(ws)
        await ws.send(json.dumps({"type": "listen", "state": "start", "mode": "auto"}))  # device re-listens
        for frame in opus_frames(0.0, 25):  # 1.5 s of silence: past follow-up, inside the wake window
            await ws.send(frame)
        msg = json.loads(await asyncio.wait_for(ws.recv(), 5))
        assert msg.get("command") == "idle"
        await ws.send(json.dumps({"type": "listen", "state": "start", "mode": "auto"}))  # next wake word
        for frame in opus_frames(0.0, 25):
            await ws.send(frame)
        with pytest.raises(asyncio.TimeoutError):  # 1.5 s < 3 s wake window: still listening
            await asyncio.wait_for(ws.recv(), 0.5)


async def test_device_volume_is_set_on_connect(unused_tcp_port):
    cfg = Config(device_token=TOKEN, hermes_api_key="unused", host="127.0.0.1", port=unused_tcp_port, device_volume=90)
    ready = asyncio.Event()
    task = asyncio.create_task(serve_forever(cfg, Fakes().backends(), ready))
    await ready.wait()
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}/xiaozhi/v1/",
                                      additional_headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
            await handshake(ws)
            msg = json.loads(await asyncio.wait_for(ws.recv(), 2))
    finally:
        task.cancel()
    assert msg["type"] == "mcp" and msg["payload"]["method"] == "tools/call"
    assert msg["payload"]["params"] == {"name": "self.audio_speaker.set_volume", "arguments": {"volume": 90}}


async def test_keepalive_reaches_idle_device(bridge, monkeypatch):
    monkeypatch.setattr(server, "KEEPALIVE_S", 0.2)
    url = await bridge(Fakes())
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
        await handshake(ws)
        msg = json.loads(await asyncio.wait_for(ws.recv(), 2))
    assert {k: msg[k] for k in ("type", "command")} == {"type": "system", "command": "keepalive"}


def announce(url: str, text: str, token: str = TOKEN) -> httpx.Response:
    base = url.split("/xiaozhi")[0].replace("ws://", "http://")
    return httpx.get(f"{base}/announce", params={"text": text}, headers={"Authorization": f"Bearer {token}"})


async def test_announce_speaks_on_connected_device(bridge):
    fakes = Fakes()
    url = await bridge(fakes)
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
        await handshake(ws)
        response = await asyncio.to_thread(announce, url, "早晨。今日會落雨，記得帶遮。")
        assert response.status_code == 202
        texts, audio = await collect_turn(ws)
    kinds = [(m["type"], m.get("state")) for m in texts]
    assert kinds == [("llm", None), ("tts", "start"), ("tts", "sentence_start"), ("tts", "sentence_start"),
                     ("tts", "stop")]
    assert fakes.spoken == ["早晨。", "今日會落雨，記得帶遮。"] and fakes.asked == [] and audio


async def test_announce_without_device_is_503(bridge):
    url = await bridge(Fakes())
    assert (await asyncio.to_thread(announce, url, "早晨")).status_code == 503


async def test_announce_while_device_busy_is_409(bridge):
    gate = asyncio.Event()
    url = await bridge(Fakes(hermes_gate=gate))
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
        await handshake(ws)
        await talk(ws)
        while not isinstance(msg := await asyncio.wait_for(ws.recv(), 5), str) or json.loads(msg)["type"] != "stt":
            pass  # wait until the turn is running
        assert (await asyncio.to_thread(announce, url, "早晨")).status_code == 409
        gate.set()


async def test_announce_bad_token_is_401(bridge):
    url = await bridge(Fakes())
    assert (await asyncio.to_thread(announce, url, "早晨", token="wrong")).status_code == 401


async def test_short_noise_burst_is_discarded_before_stt(bridge):
    fakes = Fakes()
    url = await bridge(fakes)
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
        await handshake(ws)
        await ws.send(json.dumps({"type": "listen", "state": "start", "mode": "auto"}))
        # 180 ms burst (Opus look-ahead smears ~1 frame): starts the VAD but stays under min_voiced_ms 300
        for frame in opus_frames(0.0, 3) + opus_frames(0.3, 3) + opus_frames(0.0, 20):
            await ws.send(frame)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ws.recv(), 1)
    assert fakes.stt_bytes == 0 and fakes.asked == []
