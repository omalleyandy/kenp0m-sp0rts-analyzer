"""Placeholder tests for kenp0m-sp0rts-analyzer."""

from kenp0m_sp0rts_analyzer import __version__


def test_version() -> None:
    """Test that version is defined."""
    assert __version__ == "0.1.0"


def test_version_format() -> None:
    """Test that version follows semver format."""
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
