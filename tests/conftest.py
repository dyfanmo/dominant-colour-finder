"""Shared pytest fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir() -> Path:
    return Path(__file__).parent / "test_data"
