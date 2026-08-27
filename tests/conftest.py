"""
tests/conftest.py — shared pytest fixtures for the whole test suite.

Why this file exists:
    Tests must be deterministic regardless of what's in a developer's
    local .env file. USE_HYBRID_RETRIEVAL defaults to False in
    settings.py, but if you've set USE_HYBRID_RETRIEVAL=true in your own
    .env to manually try hybrid retrieval (see app/config/settings.py),
    that value would otherwise leak straight into the test suite —
    breaking every test that assumes generate() takes the dense-only
    path (retrieve()) unless it explicitly says otherwise.

    This fixture resets that setting to a known baseline (False) before
    every test, so the suite passes identically whether or not hybrid
    retrieval is enabled in your local environment.

Tests that specifically want to exercise the hybrid path (see
test_response_generator_hybrid_flag.py) aren't affected by this — they
patch app.generation.response_generator.settings directly with a
MagicMock, which fully replaces the settings object for the duration of
that test and takes priority over this fixture.
"""
import pytest

from app.config.settings import settings


@pytest.fixture(autouse=True)
def _reset_hybrid_retrieval_flag(monkeypatch):
    """
    Runs before every test automatically. Forces settings.USE_HYBRID_RETRIEVAL
    to False so the test suite never depends on the developer's local .env.
    """
    monkeypatch.setattr(settings, "USE_HYBRID_RETRIEVAL", False)