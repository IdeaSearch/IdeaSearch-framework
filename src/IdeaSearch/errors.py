"""
Typed exceptions raised by IdeaSearch.

The framework historically used the bare built-in ``exit()`` to terminate the
process when an internal invariant was violated. ``exit()`` raises
``SystemExit``, which is *not* a subclass of ``Exception`` and therefore
slips past any ``except Exception`` block in caller code. Embedding hosts
(such as Magnus blueprints) consequently could not surface those failures
through their normal error-reporting paths — the process just died with the
diary file as the only forensic trail.

These exception types replace those ``exit()`` calls with structured raises
so downstream callers can catch ``IdeaSearchError`` (or one of its
subclasses) reliably. Messages are intentionally kept in English and are
*not* routed through gettext: they are aimed at programmers diagnosing a
crash, while the user-facing diary entries continue to be translated as
before.
"""


__all__ = [
    "IdeaSearchError",
    "IdeaSearchInternalError",
    "SamplerThreadError",
]


class IdeaSearchError(Exception):
    """Base class for all IdeaSearch-raised errors that downstream code may
    wish to catch. Always a subclass of ``Exception`` (never ``BaseException``
    directly), so a plain ``except Exception`` in caller code is sufficient.
    """


class IdeaSearchInternalError(IdeaSearchError):
    """Signals a broken internal invariant — a state the framework's own
    code paths should have prevented. Encountering one indicates a bug in
    IdeaSearch rather than misuse by the caller. Diary entries describing
    the violation are written immediately before the raise; this exception
    carries a short English summary suitable for tracebacks.
    """


class SamplerThreadError(IdeaSearchError):
    """Wraps an exception raised inside a sampler worker thread so the main
    thread can surface it to the caller of :meth:`IdeaSearcher.run`. The
    original exception is attached via ``raise ... from`` and is therefore
    available on ``__cause__``.
    """
