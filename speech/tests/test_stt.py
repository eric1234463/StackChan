from stackchan_speech.stt import DEFAULT_PROMPT, is_prompt_echo


def test_prompt_read_back_is_flagged():
    for echoed in ("英文詞保持英文：Home Assistant、Hermes、StackChan、Server、log、Docker、42C。",
                   "英文詞保持英文：Hermes、StackChan、Server、log、Docker、42C。"):  # seen 2026-11-02
        assert is_prompt_echo(echoed, DEFAULT_PROMPT)


def test_real_commands_mentioning_prompt_terms_pass():
    for said in ("幫我睇下 Home Assistant log", "幫我熄廁所燈。", "開書房燈。", "下一架42C要幾耐?", "幫我 restart 個 Docker",
                 "開空氣清新機。", "熄顯示器掛燈。", "閂客廳窗簾。"):
        assert not is_prompt_echo(said, DEFAULT_PROMPT)
