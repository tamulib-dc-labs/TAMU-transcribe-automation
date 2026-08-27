"""Word tokenisation and normalisation helpers.

Kept dependency-free (stdlib only) so the pure-logic stages are importable and
testable without torch installed.
"""

from __future__ import annotations

import re
import unicodedata

# Ranges where text is written without spaces, so one character == one "word"
# for timestamping purposes.
_CJK_RANGES = (
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK unified
    (0xF900, 0xFAFF),  # CJK compatibility
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0x20000, 0x2FA1F),  # CJK ext B+
)

_WS_RE = re.compile(r"\s+")


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def split_words(text: str) -> list[str]:
    """Split ``text`` into timestampable units.

    Whitespace-delimited tokens for space-separated scripts; one unit per
    character for CJK/Hangul, which are written without spaces. Punctuation
    stays attached to the token it follows so subtitles read naturally.
    """
    words: list[str] = []
    for chunk in _WS_RE.split(text.strip()):
        if not chunk:
            continue
        buf: list[str] = []
        for ch in chunk:
            if is_cjk(ch):
                if buf:
                    words.append("".join(buf))
                    buf = []
                words.append(ch)
            elif words and not buf and _is_punct(ch):
                # Trailing punctuation after a CJK char sticks to that char.
                words[-1] += ch
            else:
                buf.append(ch)
        if buf:
            words.append("".join(buf))
    return words


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def join_words(words: list[str]) -> str:
    """Re-join word units, omitting the space where the script does not use one."""
    out = ""
    for word in words:
        if not out:
            out = word
        elif _no_space_between(out[-1], word[0]):
            out += word
        else:
            out += " " + word
    return out


def _no_space_between(left: str, right: str) -> bool:
    if is_cjk(left) or is_cjk(right):
        return True
    return _is_punct(right) and right not in "([{\"'"


SENTENCE_ENDERS = frozenset(".!?…。！？；;")


def ends_sentence(word: str) -> bool:
    return bool(word) and word[-1] in SENTENCE_ENDERS


def normalize_for_scoring(word: str) -> str:
    """Casefold and strip punctuation, for reference-based WER scoring."""
    cleaned = "".join(
        ch for ch in unicodedata.normalize("NFKC", word) if not _is_punct(ch) or ch == "'"
    )
    return cleaned.strip("'").casefold()


def normalize_for_alignment(word: str) -> str:
    """Casefold and drop everything that is not a letter, digit or apostrophe."""
    out = []
    for ch in unicodedata.normalize("NFKC", word).casefold():
        if ch.isalnum() or ch == "'":
            out.append(ch)
    return "".join(out)


__all__ = [
    "is_cjk",
    "split_words",
    "join_words",
    "ends_sentence",
    "SENTENCE_ENDERS",
    "normalize_for_scoring",
    "normalize_for_alignment",
]
