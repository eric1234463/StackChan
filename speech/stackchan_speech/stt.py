"""Wyoming ASR server wrapping mlx-whisper (Cantonese fine-tune, Apple Silicon)."""
from __future__ import annotations

import argparse
import asyncio
import difflib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger(__name__)
RATE = 16000
DEFAULT_MODEL = "WinsonLi/whisper-large-v3-cantonese-mlx-4bit"
DEFAULT_PROMPT = (  # areas and devices as named in Home Assistant (listed via Hermes 2026-11-02)
    "香港廣東話口語。幫我開燈、熄燈、閂窗簾：客廳、書房、主人房、廚房、廁所、走廊；"
    "燈槽、顯示器掛燈、風扇、循環扇、空氣清新機、加濕器、驅蚊器、冷氣、電視、門鎖、掃地機有冇着？"
    "英文詞保持英文：Home Assistant、Hermes、StackChan、server、log、Docker、42C。"
)
# Whisper's own decode-failure limits (mlx_whisper defaults). On noise every temperature
# fails them and the last attempt is garbage such as "阿阿阿…", so those segments are dropped.
MAX_COMPRESSION_RATIO = 2.4
MIN_AVG_LOGPROB = -1.0
# On steady noise whisper can also read back its own initial prompt verbatim.
MAX_PROMPT_COVERAGE = 0.8
# Echoes seen are 20+ characters; a short command naming one device ("開空氣清新機") is mostly prompt text too.
MIN_ECHO_CHARS = 10


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).lower()


def prompt_coverage(text: str, prompt: str) -> float:
    """Share of the text made of runs (>= 4 chars) copied from the prompt."""
    a, b = _norm(text), _norm(prompt)
    if not a or not b:
        return 0.0
    blocks = difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    return sum(m.size for m in blocks if m.size >= 4) / len(a)


def is_prompt_echo(text: str, prompt: str) -> bool:
    return len(_norm(text)) >= MIN_ECHO_CHARS and prompt_coverage(text, prompt) >= MAX_PROMPT_COVERAGE

# MLX binds GPU streams to the thread that first uses them, so every model call
# goes through this one worker thread; it also serialises requests.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def _transcribe(model: str, audio: np.ndarray, language: str, prompt: str) -> str:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=model,
        language=language,
        initial_prompt=prompt or None,
        condition_on_previous_text=False,
        verbose=None,
    )
    kept = []
    for seg in result["segments"]:
        if (seg["compression_ratio"] > MAX_COMPRESSION_RATIO or seg["avg_logprob"] < MIN_AVG_LOGPROB
                or is_prompt_echo(seg["text"], prompt)):
            _LOGGER.info("dropped segment cr=%.2f logprob=%.2f: %s",
                         seg["compression_ratio"], seg["avg_logprob"], seg["text"][:40])
        else:
            kept.append(seg["text"])
    return "".join(kept).strip()


def make_info(model: str, language: str) -> Info:
    attribution = Attribution(name="WinsonLi", url=f"https://huggingface.co/{model}")
    return Info(asr=[AsrProgram(
        name="mlx-whisper", description="mlx-whisper Cantonese large-v3", attribution=attribution,
        installed=True, version="0.4", models=[AsrModel(
            name=model, description=model, attribution=attribution,
            installed=True, version="1", languages=[language])],
    )])


class SttHandler(AsyncEventHandler):
    def __init__(self, *args, model: str, language: str, prompt: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model, self.language, self.prompt = model, language, prompt
        self.converter = AudioChunkConverter(rate=RATE, width=2, channels=1)
        self.pcm = bytearray()

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(make_info(self.model, self.language).event())
        elif Transcribe.is_type(event.type):
            self.pcm.clear()
        elif AudioChunk.is_type(event.type):
            self.pcm += self.converter.convert(AudioChunk.from_event(event)).audio
        elif AudioStop.is_type(event.type):
            audio = np.frombuffer(bytes(self.pcm), np.int16).astype(np.float32) / 32768.0
            self.pcm.clear()
            started = time.monotonic()
            text = await asyncio.get_running_loop().run_in_executor(
                _EXECUTOR, _transcribe, self.model, audio, self.language, self.prompt)
            _LOGGER.info("transcribed %.1fs audio in %.2fs: %s",
                         len(audio) / RATE, time.monotonic() - started, text)
            await self.write_event(Transcript(text=text).event())
            return False
        return True


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local MLX model dir")
    parser.add_argument("--language", default="yue")
    parser.add_argument("--initial-prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    _LOGGER.info("Loading %s", args.model)
    await asyncio.get_running_loop().run_in_executor(
        _EXECUTOR, _transcribe, args.model, np.zeros(RATE, np.float32), args.language, "")
    _LOGGER.info("Model ready on %s", args.uri)
    await AsyncServer.from_uri(args.uri).run(lambda *a, **k: SttHandler(
        *a, model=args.model, language=args.language, prompt=args.initial_prompt, **k))


if __name__ == "__main__":
    asyncio.run(main())
