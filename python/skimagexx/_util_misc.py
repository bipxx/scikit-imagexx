"""Warning metadata and documentation search."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _canonical_pending_skimage2_change():
    try:
        from skimage.util import PendingSkimage2Change
    except ModuleNotFoundError as error:
        if error.name != 'skimage':
            raise

        class PendingSkimage2Change(PendingDeprecationWarning):
            """A warning about API usage that will silently change or break in skimage2."""

            __module__ = f'{__package__}.util'
            __qualname__ = 'PendingSkimage2Change'

    # Preserve warning filters and pickle identity across calls to either backend.
    return PendingSkimage2Change


def __getattr__(name):
    if name == 'PendingSkimage2Change':
        return _canonical_pending_skimage2_change()
    raise AttributeError(name)


@lru_cache(maxsize=1)
def _search_entries():
    from skimage._vendored.numpy_lookfor import (
        _function_signature_re,
        _lookfor_generate_cache,
    )

    # Use the installed reference's corpus, including private and optional
    # modules discovered by its documentation loader.
    entries = []
    for name, (doc, kind, index) in _lookfor_generate_cache('skimage', True, False).items():
        if kind in ('module', 'object'):
            continue
        normalized = doc.lower()
        lead = '\n'.join(normalized.strip().split('\n', 3)[:3])
        lines = (line.strip() for line in doc.split('\n') if line.strip())
        summary = next(lines, '')
        if _function_signature_re.search(summary):
            summary = next(lines, '')
        kind_score = 1000 if kind in ('func', 'class') else -1000
        base_score = -len(name) * 5 - name.count('.') * 10 + kind_score
        order_score = max(-index / 100, -100)
        entries.append((normalized, lead, name, f'{name}\n    {summary}', base_score, order_score))
    return tuple(entries)


def lookfor(what):
    """Do a keyword search on scikit-image docstrings and print results."""
    try:
        entries = _search_entries()
    except ImportError:
        from ._compat import fallback

        return fallback('util', 'lookfor', what)

    words = str(what).lower().split()
    if not words:
        return
    first, *rest = words
    hits = []
    for normalized, lead, name, summary, base, order in entries:
        if first in normalized and (not rest or all(word in normalized for word in rest)):
            score = base + 200 * sum(word in lead for word in words)
            score += 30 * sum(word in name for word in words)
            hits.append((score + order, summary))

    heading = f"Search results for '{' '.join(words)}'"
    hits.sort(key=lambda hit: hit[0])
    lines = [heading, '-' * len(heading)]
    lines.extend(summary for _, summary in reversed(hits))
    if not hits:
        lines.append('Nothing found.')
    output = '\n'.join(lines)
    if len(lines) > 10:
        import pydoc

        pydoc.getpager()(output)
    else:
        print(output)
