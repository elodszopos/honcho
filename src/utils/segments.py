"""Split a query into the thoughts it carries: one per line, then one per sentence."""

import warnings

from src.config import settings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import pysbd

_segmenter = pysbd.Segmenter(language="en", clean=False)


def split_thoughts(text: str, *, min_chars: int | None = None) -> list[str]:
    """Lines split into sentences; a fragment under ``min_chars`` joins the segment before it."""
    floor = settings.EMBEDDING.THOUGHT_MIN_CHARS if min_chars is None else min_chars
    segments: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        for sentence in _segmenter.segment(line):
            sentence = sentence.strip()
            if not sentence:
                continue
            if segments and len(sentence) < floor:
                segments[-1] = f"{segments[-1]} {sentence}"
            else:
                segments.append(sentence)
    if not segments:
        stripped = (text or "").strip()
        return [stripped] if stripped else []
    return segments
