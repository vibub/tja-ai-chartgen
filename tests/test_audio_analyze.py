import pytest

from tja_ai_chartgen.audio.analyze import normalize_bpm


def test_normalize_bpm_keeps_taiko_friendly_range():
    assert normalize_bpm(60) == 120
    assert normalize_bpm(260) == 130


def test_normalize_bpm_rejects_non_positive_values():
    with pytest.raises(ValueError, match="BPM must be positive"):
        normalize_bpm(0)
