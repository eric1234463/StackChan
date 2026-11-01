"""Minimal Wyoming TTS server wrapping sherpa-onnx Cantonese VITS."""
import argparse
import asyncio
import logging
import socket
from urllib.parse import urlparse

import numpy as np
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

_LOGGER = logging.getLogger(__name__)
_TTS = None
_TTS_LOCK = asyncio.Lock()

VOICE_NAME = "yue_HK-xiaomaiiwn-medium"


def load_tts(model_dir: str):
    global _TTS
    if _TTS is not None:
        return _TTS
    import sherpa_onnx
    _TTS = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=f"{model_dir}/vits-cantonese-hf-xiaomaiiwn.onnx",
                    lexicon=f"{model_dir}/lexicon.txt",
                    tokens=f"{model_dir}/tokens.txt",
                )
            )
        )
    )
    return _TTS


def make_info() -> Info:
    return Info(
        tts=[
            TtsProgram(
                name="canto-vits",
                description="Local Cantonese VITS (sherpa-onnx)",
                attribution=Attribution(name="csukuangfj", url="https://huggingface.co/csukuangfj/vits-cantonese-hf-xiaomaiiwn"),
                installed=True,
                version="1.0",
                voices=[
                    TtsVoice(
                        name=VOICE_NAME,
                        description="Cantonese female (xiaomaiiwn)",
                        attribution=Attribution(name="csukuangfj", url=""),
                        installed=True,
                        version="1.0",
                        languages=["yue", "zh-HK", "zh"],
                    )
                ],
            )
        ],
    )


class CantoEventHandler(AsyncEventHandler):
    def __init__(self, *args, model_dir: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_dir = model_dir

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(make_info().event())
            return True
        if Synthesize.is_type(event.type):
            synth = Synthesize.from_event(event)
            async with _TTS_LOCK:
                tts = await asyncio.to_thread(load_tts, self.model_dir)
                audio = await asyncio.to_thread(
                    tts.generate, synth.text, 0, 1.0
                )
            rate = audio.sample_rate
            samples = (np.array(audio.samples) * 32767).astype(np.int16)
            await self.write_event(
                AudioStart(rate=rate, width=2, channels=1).event()
            )
            await self.write_event(
                AudioChunk(
                    audio=samples.tobytes(), rate=rate, width=2, channels=1
                ).event()
            )
            await self.write_event(AudioStop().event())
            return True
        return False


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10201")
    parser.add_argument("--model-dir", required=True,
                        help="Directory with vits-cantonese-hf-xiaomaiiwn.onnx, lexicon.txt, tokens.txt")
    parser.add_argument("--zeroconf", nargs="?", const="canto-tts",
                        help="Advertise via zeroconf for HA auto-discovery")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    _LOGGER.info("Preloading Cantonese TTS model...")
    await asyncio.to_thread(load_tts, args.model_dir)
    _LOGGER.info("Model ready")
    zc = None
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
    server = AsyncServer.from_uri(args.uri)
    await server.run(
        lambda *a, **k: CantoEventHandler(*a, model_dir=args.model_dir, **k)
    )


if __name__ == "__main__":
    asyncio.run(main())
