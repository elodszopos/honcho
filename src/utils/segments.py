"""Split a query into the thoughts it carries: one per line, then one per sentence."""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import pysbd

MIN_SEGMENT_CHARS = 20
MAX_SEGMENTS = 12

_segmenter = pysbd.Segmenter(language="en", clean=False)


def split_thoughts(
    text: str, *, min_chars: int = MIN_SEGMENT_CHARS, max_segments: int = MAX_SEGMENTS
) -> list[str]:
    """Lines split into sentences; a fragment under ``min_chars`` joins the segment before it."""
    segments: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        for sentence in _segmenter.segment(line):
            sentence = sentence.strip()
            if not sentence:
                continue
            if segments and len(sentence) < min_chars:
                segments[-1] = f"{segments[-1]} {sentence}"
            else:
                segments.append(sentence)
    if not segments:
        stripped = (text or "").strip()
        return [stripped] if stripped else []
    return segments[:max_segments]
