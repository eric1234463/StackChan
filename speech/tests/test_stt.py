from stackchan_speech.stt import DEFAULT_PROMPT, MAX_PROMPT_COVERAGE, prompt_coverage


def test_prompt_read_back_is_flagged():
    for echoed in ("英文詞保持英文：Home Assistant、Hermes、StackChan、Server、log、Docker、42C。",
                   "英文詞保持英文：Hermes、StackChan、Server、log、Docker、42C。"):  # seen 2026-11-02
        assert prompt_coverage(echoed, DEFAULT_PROMPT) >= MAX_PROMPT_COVERAGE


def test_real_commands_mentioning_prompt_terms_pass():
    for said in ("幫我睇下 Home Assistant log", "幫我熄廁所燈。", "下一架42C要幾耐?", "幫我 restart 個 Docker"):
        assert prompt_coverage(said, DEFAULT_PROMPT) < MAX_PROMPT_COVERAGE
