"""Reply text cleanup and sentence splitting for TTS."""
from __future__ import annotations

import re

_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_URL = re.compile(r"https?://\S+")
_MARKDOWN = re.compile(r"[*_`#>|~\[\]]")
_LIST_MARK = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.M)
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF‍️]"
)
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;\n])")
_SOFT_BREAK = re.compile(r"(?<=[，,、])")


def clean(text: str) -> str:
    text = _CODE_BLOCK.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _LIST_MARK.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def split_sentences(text: str, max_chars: int = 40) -> list[str]:
    out: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            out.append(sentence)
            continue
        chunk = ""
        for part in _SOFT_BREAK.split(sentence):
            if chunk and len(chunk) + len(part) > max_chars:
                out.append(chunk)
                chunk = ""
            chunk += part
            while len(chunk) > max_chars:
                out.append(chunk[:max_chars])
                chunk = chunk[max_chars:]
        if chunk.strip():
            out.append(chunk.strip())
    return out


_CJK = re.compile(r"[\u4e00-\u9fff]")
# Phrases whisper produces from silence/noise (subtitle credits seen in its training data).
_HALLUCINATIONS = ("字幕", "amara", "subscribe", "訂閱", "thanks for watching", "thank you for watching",
                   "ming pao", "謝謝觀賞", "多謝收看", "多謝收睇", "多謝您收睇")


def is_meaningful(transcript: str, romanized: bool = False) -> bool:
    """Drop empty, filler-only, non-Chinese or known-hallucination transcripts.

    romanized: the STT writes Liujgoj romanization, so Latin text counts as Cantonese.
    """
    core = re.sub(r"[\s\W_]+", "", transcript)
    lowered = transcript.lower()
    if any(h in lowered for h in _HALLUCINATIONS):
        return False
    if _CJK.search(transcript) or not romanized:
        return len(core) >= 2 and bool(_CJK.search(transcript))
    # SHORTCUT: English-only hallucinations also pass here; none seen in 171 device clips with the Liujgoj model.
    words = re.findall(r"[a-z]+", lowered)
    # On noise the model says one short syllable ("Aaj!", "Gok.") or repeats one ("Heij! Heij! ...").
    return len(set(words)) >= 2 or (len(words) == 1 and len(words[0]) >= 6)


# Replies that only acknowledge ("OK", "收到", "好，冇問題"): in a follow-up window they end the conversation.
_CLOSING = sorted({"ok", "okay", "收到", "好", "好嘅", "冇問題", "多謝", "唔該", "明白", "明白晒", "知道", "得",
                   "冇嘢", "冇事", "唔使", "係咁先", "拜拜", "bye", "thankyou", "thanks"}, key=len, reverse=True)
_PARTICLES = "喇啦呀啊囉嘅吖喎咯了"
# The same phrases and particles in Liujgoj romanization, normalised by _toneless ("Mxgoij" -> "mgoi").
_CLOSING_ROMAN = sorted({"saudou", "hou", "houge", "moumantai", "doze", "mgoi", "mingbaak", "mingbaaksaai", "zidou",
                         "dak", "mouje", "mousi", "msai", "haigamsin", "baaibaai"}, key=len, reverse=True)
_PARTICLES_ROMAN = ("laa", "la", "aa", "a", "lo", "ge", "wo", "gaa")


def _toneless(text: str) -> str:
    """Liujgoj -> toneless Jyutping-like letters: tone letters j/r/x/q and syllable-final h dropped, y- -> j-."""
    s = re.sub(r"[^a-z]", "", text.lower())
    s = re.sub(r"[jrxq]", "", s)
    s = re.sub(r"h(?![aeiou])", "", s)
    return s.replace("y", "j")


def _only_phrases(rest: str, phrases: list[str], strip) -> bool:
    if not rest:
        return False
    while rest:
        rest = strip(rest)
        phrase = next((p for p in phrases if rest.startswith(p)), None)
        if phrase is None:
            return rest == ""
        rest = rest[len(phrase):]
    return True


def _strip_roman_particles(rest: str) -> str:
    while (p := next((p for p in _PARTICLES_ROMAN if rest.startswith(p)), None)):
        rest = rest[len(p):]
    return rest


def is_closing(transcript: str, romanized: bool = False) -> bool:
    if _only_phrases(re.sub(r"[\s\W_]+", "", transcript).lower(), _CLOSING, lambda r: r.lstrip(_PARTICLES)):
        return True
    return (romanized and not _CJK.search(transcript)
            and _only_phrases(_toneless(transcript), _CLOSING_ROMAN, _strip_roman_particles))
