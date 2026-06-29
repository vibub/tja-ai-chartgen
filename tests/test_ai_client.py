import json

from tja_ai_chartgen.ai.client import generate_chart_bars_with_ai, sanitize_ai_bars
from tja_ai_chartgen.tja.model import BarFeature, ChartBar, SongAnalysis


def test_sanitize_ai_bars_normalizes_count_length_and_characters():
    bars = [
        ChartBar(index=10, notes="12x"),
        ChartBar(index=11, notes="12340123401234012340"),
    ]

    sanitized = sanitize_ai_bars(bars, expected_count=3)

    assert [bar.index for bar in sanitized] == [0, 1, 2]
    assert sanitized[0].notes == "1200000000000000"
    assert sanitized[1].notes == "1234012340123401"
    assert sanitized[2].notes == "1000100010001000"


def test_generate_chart_bars_with_ai_parses_litellm_dict_response(monkeypatch):
    analysis = SongAnalysis(
        title="Song Title",
        artist=None,
        audio_file="song.mp3",
        ogg_file="song.ogg",
        bpm=120,
        offset=0,
        bars=[BarFeature(index=0, start_time=0, end_time=2, energy=0.5)],
    )
    payload = {"bars": [{"bar": 1, "notes": "1000100010001000"}]}

    def fake_completion(model, messages, temperature):
        assert model == "fake/model"
        assert messages[0]["role"] == "user"
        assert temperature == 0.7
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, raw = generate_chart_bars_with_ai(analysis, "Oni", 10, "technical", model="fake/model")

    assert raw == payload
    assert bars == [ChartBar(index=0, notes="1000100010001000")]
