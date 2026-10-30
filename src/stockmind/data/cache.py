"""Automatic response cache for data sources.

Why a cache?
    A cold fetch from PSX takes several seconds. Without a cache, every user
    request would wait for the same download again, and PSX would see many
    identical requests. Data libraries and web apps solve this the same way:
    psxdata (~/.psxdata/cache), yfinance (requests-cache), Hugging Face
    (~/.cache/huggingface), and Flask apps (Flask-Caching / Redis).

How it behaves
    - Lives in the user's home folder (~/.stockmind/cache), not in the
      project, and is never edited by hand. Deleting it is always safe:
      the agent simply fetches again.
    - Every entry remembers when it was fetched. Entries older than their
      max age are fetched again on the next request.
    - If the source fails, a previously fetched (stale) value is returned
      with a warning instead of failing the whole request.

Built on `diskcache`, a widely used, thread-safe cache library that is
already installed as a dependency of psxdata.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import diskcache


@dataclass
class CachedResult:
    """A value plus where it came from."""

    value: Any
    fetched_at: datetime
    warning: str | None = None


class ResponseCache:
    """Time-limited cache with a stale-if-error fallback."""

    def __init__(self, directory: Path | str) -> None:
        self._store = diskcache.Cache(str(directory))

    def get_or_fetch(
        self,
        key: str,
        max_age: timedelta,
        fetch: Callable[[], Any],
        fallback_errors: tuple[type[Exception], ...],
        force_refresh: bool = False,
    ) -> CachedResult:
        """
        Return a fresh cached value, or call `fetch` and cache its result.

        Raises:
            The fetch error, if fetching fails and nothing is cached.
        """
        cached = self._store.get(key)  # (fetched_at, value) or None

        if cached and not force_refresh:
            fetched_at, value = cached
            if datetime.now() - fetched_at <= max_age:
                return CachedResult(value, fetched_at)

        try:
            value = fetch()
        except fallback_errors as exc:
            if not cached:
                raise
            fetched_at, value = cached
            return CachedResult(
                value,
                fetched_at,
                warning=(
                    f"{key}: source unavailable ({type(exc).__name__}: {exc}); "
                    f"using data fetched at {fetched_at:%Y-%m-%d %H:%M}."
                ),
            )

        fetched_at = datetime.now()
        self._store.set(key, (fetched_at, value))
        return CachedResult(value, fetched_at)

    def clear(self) -> None:
        """Remove every cached entry."""
        self._store.clear()
