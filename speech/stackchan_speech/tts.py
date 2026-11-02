"""Wyoming TTS server wrapping Qwen3-TTS 1.7B + HK Cantonese LoRA (merged, MLX 8-bit)."""
import argparse
import asyncio
import logging
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import numpy as np
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

_LOGGER = logging.getLogger(__name__)
_MODEL = None
# MLX binds GPU streams to the thread that first uses them; one worker also serialises requests.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

RATE = 24000
DEFAULT_VOICE = "hk_voice_343"  # the adapter's voice; a voice trained with voice/train/train.sh adds e.g. "eric"
MAX_CHUNK_CHARS = 45  # model card: longer inputs drop words; split at punctuation
LEAD_KEEP_S, TAIL_KEEP_S = 0.04, 0.15  # every generation starts with 0.5-1.2 s of silence
SECONDS_PER_CHAR_MAX = 0.4  # measured ~0.25-0.3 s per Cantonese character
MAX_ATTEMPTS = 3
MAX_PAUSE_S = 0.22  # conversational comma pause, before --speed (0.18 s at 1.2x)
PAUSE_MARGIN_DB = 4.0  # pause = within this of the noise floor (trim uses 10 dB)


def load(model_dir: str):
    global _MODEL
    if _MODEL is None:
        from mlx_audio.tts.utils import load_model
        _MODEL = load_model(model_dir)
    return _MODEL


def chunks(text: str) -> list[str]:
    """Split at sentence punctuation, then at commas, into pieces of at most MAX_CHUNK_CHARS."""
    out: list[str] = []
    for sentence in re.findall(r"[^。！？!?\n]+[。！？!?]*", text):
        piece = ""
        for clause in re.findall(r"[^，、；,;]+[，、；,;]*", sentence):
            if piece and len(piece) + len(clause) > MAX_CHUNK_CHARS:
                out.append(piece)
                piece = ""
            piece += clause
            while len(piece) > MAX_CHUNK_CHARS:  # a single clause that is still too long
                out.append(piece[:MAX_CHUNK_CHARS])
                piece = piece[MAX_CHUNK_CHARS:]
        if piece.strip():
            out.append(piece)
    return [c.strip() for c in out if c.strip()]


def _voiced(audio: np.ndarray, margin_db: float = 10.0) -> tuple[np.ndarray, int]:
    """Per 20 ms frame: is it above silence? A trained voice also reproduces its recording room's hiss (~37 dB
    below peak), so "silence" is anything under margin_db above the noise floor (the 10th-percentile non-silent frame;
    every generation has >=0.5 s of lead-in), and never less than 40 dB or more than 20 dB below peak."""
    frame = int(RATE * 0.02)
    energy = np.sqrt([np.mean(audio[i:i + frame] ** 2) for i in range(0, max(len(audio) - frame, 1), frame)])
    peak = float(energy.max())
    audible = energy[energy > 3e-4]  # digital-silence stretches (exact zeros) are not the noise floor
    floor = float(np.percentile(audible, 10)) * 10 ** (margin_db / 20) if audible.size else 0.0
    return energy > max(min(floor, peak * 0.1), peak * 0.01, 1e-4), frame


def trim(audio: np.ndarray) -> np.ndarray:
    """Cut the generated lead-in/trailing silence."""
    is_voiced, frame = _voiced(audio)
    voiced = np.where(is_voiced)[0]
    if voiced.size == 0:
        return audio[:0]
    start = max(voiced[0] * frame - int(RATE * LEAD_KEEP_S), 0)
    end = min((voiced[-1] + 1) * frame + int(RATE * TAIL_KEEP_S), len(audio))
    return audio[start:end]


def cap_pauses(audio: np.ndarray) -> np.ndarray:
    """Shorten internal silences to MAX_PAUSE_S (keeping both edges of each pause): the eric voice copies the
    400-680 ms comma pauses of read-aloud takes, which sound like stalls in conversation. Stricter than trim
    (PAUSE_MARGIN_DB) so breath tails and fricatives (~8 dB above the hiss) are never cut; splices crossfade."""
    if not len(audio):
        return audio
    is_voiced, frame = _voiced(audio, PAUSE_MARGIN_DB)
    keep = int(RATE * MAX_PAUSE_S) // 2
    out, pos, i = [], 0, 0
    while i < len(is_voiced):
        if is_voiced[i]:
            i += 1
            continue
        j = i
        while j < len(is_voiced) and not is_voiced[j]:
            j += 1
        if 0 < i and j < len(is_voiced) and (j - i) * frame > 2 * keep:  # internal pause only
            out += [audio[pos:i * frame + keep]]
            pos = j * frame - keep
        i = j
    out.append(audio[pos:])
    fade = int(RATE * 0.01)
    joined = out[0]
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    for seg in out[1:]:
        joined = np.concatenate([joined[:-fade], joined[-fade:] * (1 - ramp) + seg[:fade] * ramp, seg[fade:]])
    return joined


def time_stretch(audio: np.ndarray, speed: float) -> np.ndarray:
    """WSOLA: play `speed` times faster at the same pitch. 40 ms Hann frames at 50 % overlap; each input frame is
    taken within +-12 ms of its nominal position where it best continues the previous one (no phase smearing)."""
    if speed == 1.0 or len(audio) < 2 * int(RATE * 0.04):
        return audio
    frame, tol = int(RATE * 0.04), int(RATE * 0.012)
    hop = frame // 2
    window = np.hanning(frame).astype(np.float32)
    n_out = int(len(audio) / speed)
    padded = np.concatenate([np.zeros(tol, np.float32), audio, np.zeros(frame + tol, np.float32)])
    out = np.zeros(n_out + frame, np.float32)
    norm = np.zeros(n_out + frame, np.float32)
    prev = tol  # where the previous frame was taken from (in padded coordinates)
    for k in range(0, n_out, hop):
        nominal = tol + int(k * speed)
        target = padded[prev + hop:prev + hop + frame]  # natural continuation of the previous frame
        region = padded[nominal - tol:nominal + tol + frame]
        best = int(np.argmax(np.correlate(region, target, "valid"))) if k else tol
        prev = nominal - tol + best
        out[k:k + frame] += padded[prev:prev + frame] * window
        norm[k:k + frame] += window
    return out[:n_out] / np.maximum(norm[:n_out], 1e-3)


def synth_chunk(model_dir: str, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
    model = load(model_dir)
    # Sampling sometimes keeps talking after the text ("收到，幫緊你" + 3 s of babble); far too long = retry.
    limit_s = len(re.sub(r"\W", "", text)) * SECONDS_PER_CHAR_MAX + 1.0
    best = None
    for attempt in range(MAX_ATTEMPTS):
        # cap pauses before trimming: the noise-floor estimate needs the generated lead-in hiss
        audio = trim(cap_pauses(np.concatenate([np.array(r.audio, dtype=np.float32)
                                     for r in model.generate(text=text, voice=voice, lang_code="chinese")])))
        if best is None or len(audio) < len(best):
            best = audio
        if len(audio) / RATE <= limit_s:
            break
        _LOGGER.warning("attempt %d: %.1fs audio for %d chars, retrying: %s", attempt + 1, len(audio) / RATE, len(text), text)
    return (np.clip(time_stretch(best, speed), -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def make_info(voice: str) -> Info:
    attribution = Attribution(name="jmtl", url="https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts")
    return Info(tts=[TtsProgram(
        name="qwen3-tts-hk", description="Qwen3-TTS 1.7B + HK Cantonese LoRA (MLX 8-bit)",
        attribution=attribution, installed=True, version="0.0.5",
        voices=[TtsVoice(name=voice, description=f"HK Cantonese, voice {voice} (synthetic)",
                         attribution=attribution, installed=True, version="0.0.5",
                         languages=["yue", "zh-HK", "zh"])],
    )])


class TtsHandler(AsyncEventHandler):
    def __init__(self, *args, model_dir: str, voice: str, speed: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_dir, self.voice, self.speed = model_dir, voice, speed

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(make_info(self.voice).event())
            return True
        if Synthesize.is_type(event.type):
            text = Synthesize.from_event(event).text
            loop = asyncio.get_running_loop()
            started = time.monotonic()
            await self.write_event(AudioStart(rate=RATE, width=2, channels=1).event())
            samples = 0
            for piece in chunks(text):  # stream chunk by chunk so long replies start sooner
                pcm = await loop.run_in_executor(_EXECUTOR, synth_chunk, self.model_dir, piece, self.voice,
                                                 self.speed)
                samples += len(pcm) // 2
                await self.write_event(AudioChunk(audio=pcm, rate=RATE, width=2, channels=1).event())
            await self.write_event(AudioStop().event())
            _LOGGER.info("synthesized %.1fs audio in %.2fs: %s", samples / RATE, time.monotonic() - started, text)
            return True
        return False


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10201")
    parser.add_argument("--model-dir", required=True,
                        help="Merged Qwen3-TTS HK Cantonese MLX model (apply_adapter.py + mlx_audio.convert)")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="speaker name baked into the model (spk_id)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="speaking-rate factor applied after synthesis (pitch kept); a voice trained on read-aloud "
                             "takes speaks at reading pace, ~1.15-1.25 sounds conversational")
    parser.add_argument("--zeroconf", nargs="?", const="canto-tts",
                        help="Advertise via zeroconf for HA auto-discovery")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    _LOGGER.info("Loading %s", args.model_dir)
    # warm-up; also fails fast if --voice is not in the model
    await asyncio.get_running_loop().run_in_executor(_EXECUTOR, synth_chunk, args.model_dir, "你好。", args.voice)
    _LOGGER.info("Model ready, voice %s", args.voice)
    if args.zeroconf:
        from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf
        zc = AsyncZeroconf()
        info = AsyncServiceInfo(
            "_wyoming._tcp.local.",
            f"{args.zeroconf}._wyoming._tcp.local.",
            port=urlparse(args.uri).port,
            properties={"name": "canto-tts", "version": "1.0"},
            server=f"{socket.gethostname().removesuffix('.local')}.local.",
        )
        await zc.async_register_service(info)
        _LOGGER.info("Zeroconf advertised as %s", args.zeroconf)
    await AsyncServer.from_uri(args.uri).run(
        lambda *a, **k: TtsHandler(*a, model_dir=args.model_dir, voice=args.voice, speed=args.speed, **k))


if __name__ == "__main__":
    asyncio.run(main())
