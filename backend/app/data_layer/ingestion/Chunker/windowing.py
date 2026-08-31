from typing import List, Tuple


def sliding_windows(text: str, size: int, overlap: int) -> List[Tuple[int, int]]:
    """Offsets of successive windows over text, none longer than size.

    A window is pulled back to the last whitespace it contains so words are not
    cut in half; a run with no whitespace in it (a base64 blob, a minified
    line) has nothing to pull back to and is split at exactly size.
    """
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(
            f"chunk overlap must be in [0, {size}), got {overlap}"
        )

    spans: List[Tuple[int, int]] = []
    length = len(text)
    start = 0

    while start < length:
        while start < length and text[start].isspace():
            start += 1
        if start >= length:
            break

        end = min(start + size, length)
        if end < length:
            cut = -1
            for index in range(end - 1, start, -1):
                if text[index].isspace():
                    cut = index
                    break
            if cut > start:
                end = cut

        spans.append((start, end))
        if end >= length:
            break
        # max(): a window pulled back to a word boundary can be shorter than
        # the overlap, and end - overlap would then step backwards forever.
        start = max(end - overlap, start + 1)

    return spans
