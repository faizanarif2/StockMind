"""Tests for the response cache."""

from datetime import timedelta

import pytest

from stockmind.data.cache import ResponseCache


class Source:
    """Fake data source that counts calls and can be switched off."""

    def __init__(self):
        self.calls = 0
        self.online = True

    def __call__(self):
        self.calls += 1
        if not self.online:
            raise ConnectionError("offline")
        return f"value-{self.calls}"


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path)


def test_fresh_value_is_reused(cache):
    source = Source()
    first = cache.get_or_fetch("k", timedelta(hours=1), source, (ConnectionError,))
    second = cache.get_or_fetch("k", timedelta(hours=1), source, (ConnectionError,))

    assert first.value == second.value == "value-1"
    assert source.calls == 1


def test_expired_value_is_fetched_again(cache):
    source = Source()
    cache.get_or_fetch("k", timedelta(0), source, (ConnectionError,))
    result = cache.get_or_fetch("k", timedelta(0), source, (ConnectionError,))

    assert result.value == "value-2"


def test_stale_value_is_used_when_source_fails(cache):
    source = Source()
    cache.get_or_fetch("k", timedelta(0), source, (ConnectionError,))
    source.online = False

    result = cache.get_or_fetch("k", timedelta(0), source, (ConnectionError,))

    assert result.value == "value-1"
    assert "source unavailable" in result.warning


def test_failure_without_cached_value_raises(cache):
    source = Source()
    source.online = False

    with pytest.raises(ConnectionError):
        cache.get_or_fetch("k", timedelta(hours=1), source, (ConnectionError,))


def test_unexpected_errors_are_not_hidden(cache):
    def buggy():
        raise KeyError("bug")

    with pytest.raises(KeyError):
        cache.get_or_fetch("k", timedelta(hours=1), buggy, (ConnectionError,))
