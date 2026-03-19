"""Shared pytest fixtures for Oracle V2 tests."""
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_match():
    """A minimal cricket match dict for testing."""
    return {
        "team_a": "India",
        "team_b": "New Zealand",
        "winner": "India",
        "venue": "Mumbai",
        "stage": "group",
        "sport": "cricket",
        "date": "2026-02-15",
    }


@pytest.fixture
def sample_matches():
    """A list of cricket matches for training/testing."""
    return [
        {"team_a": "India", "team_b": "Pakistan", "winner": "India", "venue": "Colombo", "stage": "group", "sport": "cricket", "margin_numeric": 61},
        {"team_a": "England", "team_b": "Nepal", "winner": "England", "venue": "Mumbai", "stage": "group", "sport": "cricket"},
        {"team_a": "New Zealand", "team_b": "Afghanistan", "winner": "New Zealand", "venue": "Chennai", "stage": "group", "sport": "cricket"},
        {"team_a": "South Africa", "team_b": "Canada", "winner": "South Africa", "venue": "Ahmedabad", "stage": "group", "sport": "cricket"},
        {"team_a": "Australia", "team_b": "Ireland", "winner": "Australia", "venue": "Colombo", "stage": "group", "sport": "cricket"},
        {"team_a": "West Indies", "team_b": "Scotland", "winner": "West Indies", "venue": "Kolkata", "stage": "group", "sport": "cricket"},
        {"team_a": "Sri Lanka", "team_b": "Ireland", "winner": "Sri Lanka", "venue": "Colombo", "stage": "group", "sport": "cricket"},
        {"team_a": "Zimbabwe", "team_b": "Oman", "winner": "Zimbabwe", "venue": "Colombo", "stage": "group", "sport": "cricket"},
        {"team_a": "India", "team_b": "South Africa", "winner": "South Africa", "venue": "Ahmedabad", "stage": "super8", "sport": "cricket", "margin_numeric": 76},
        {"team_a": "England", "team_b": "New Zealand", "winner": "England", "venue": "Colombo", "stage": "super8", "sport": "cricket"},
        {"team_a": "India", "team_b": "West Indies", "winner": "India", "venue": "Kolkata", "stage": "super8", "sport": "cricket"},
        {"team_a": "India", "team_b": "England", "winner": "India", "venue": "Mumbai", "stage": "semi", "sport": "cricket", "margin_numeric": 7},
    ]


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database for testing."""
    from core import OracleDB
    db_path = tmp_path / "test_oracle.db"
    return OracleDB(str(db_path))
