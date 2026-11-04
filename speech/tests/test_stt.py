from stackchan_speech.stt import DEFAULT_PROMPT, is_prompt_echo


def test_prompt_read_back_is_flagged():
    for echoed in ("英文詞保持英文：Home Assistant、Hermes、StackChan、Server、log、Docker、42C。",
                   "英文詞保持英文：Hermes、StackChan、Server、log、Docker、42C。"):  # seen 2026-11-02
        assert is_prompt_echo(echoed, DEFAULT_PROMPT)


def test_real_commands_mentioning_prompt_terms_pass():
    for said in ("幫我睇下 Home Assistant log", "幫我熄廁所燈。", "開書房燈。", "下一架42C要幾耐?", "幫我 restart 個 Docker",
                 "開空氣清新機。", "熄顯示器掛燈。", "閂客廳窗簾。"):
        assert not is_prompt_echo(said, DEFAULT_PROMPT)


def test_qwen_engine_caps_tokens_by_audio_length(monkeypatch):
    import numpy as np
    from stackchan_speech import stt

    calls = []

    class Fake:
        def generate(self, audio, **kwargs):
            calls.append(kwargs)
            return type("Out", (), {"text": " Hour. "})()

    monkeypatch.setattr(stt, "_load_qwen", lambda model: Fake())
    assert stt.ENGINES["qwen3-asr"]("dir", np.zeros(stt.RATE * 15, np.float32), "yue", "prompt") == "Hour."
    assert calls == [{"max_tokens": 15 * stt.QWEN_TOKENS_PER_S + stt.QWEN_MIN_TOKENS}]  # no initial prompt
    assert stt.make_info("dir", "yue", "qwen3-asr").asr[0].name == "mlx-audio-qwen3-asr"
    assert stt.make_info("m", "yue").asr[0].name == "mlx-whisper"
