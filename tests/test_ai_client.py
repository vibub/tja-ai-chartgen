import json

import pytest

from tja_ai_chartgen.ai.client import (
    AiOutputRepairError,
    generate_chart_bars_with_ai,
    sanitize_ai_bars,
)
from tja_ai_chartgen.ai.prompts import build_chart_generation_payload
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


def test_build_chart_generation_payload_includes_density():
    analysis = _analysis()

    payload = build_chart_generation_payload(analysis, "Oni", 10, "technical", "high")

    assert payload["density"] == "high"
    assert payload["style"] == "technical"
    assert payload["bars"][0]["grids_per_bar"] == 16


def test_generate_chart_bars_with_ai_parses_litellm_dict_response(monkeypatch):
    payload = {"bars": [{"bar": 1, "notes": "1000100010001000"}]}

    def fake_completion(model, messages, temperature):
        assert model == "fake/model"
        assert messages[0]["role"] == "user"
        assert temperature == 0.7
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, raw = generate_chart_bars_with_ai(_analysis(), "Oni", 10, "technical", model="fake/model")

    assert raw["final"] == payload
    assert raw["model"] == "fake/model"
    assert bars == [ChartBar(index=0, notes="1000100010001000")]


def test_generate_chart_bars_with_ai_passes_openai_compatible_connection_options(monkeypatch):
    payload = {"bars": [{"bar": 1, "notes": "1000100010001000"}]}

    def fake_completion(**kwargs):
        assert kwargs["model"] == "openai/custom-model"
        assert kwargs["api_base"] == "https://llm.example.com/v1"
        assert kwargs["api_key"] == "test-key"
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    _, raw = generate_chart_bars_with_ai(
        _analysis(),
        "Oni",
        10,
        "technical",
        model="openai/custom-model",
        api_base="https://llm.example.com/v1",
        api_key="test-key",
    )

    assert raw["api_base"] == "https://llm.example.com/v1"
    assert raw["api_key_provided"] is True


def test_generate_chart_bars_with_ai_reads_openai_env_names(monkeypatch):
    payload = {"bars": [{"bar": 1, "notes": "1000100010001000"}]}
    monkeypatch.setenv("MODEL", "openai/env-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    def fake_completion(**kwargs):
        assert kwargs["model"] == "openai/env-model"
        assert kwargs["api_base"] == "https://env.example.com/v1"
        assert kwargs["api_key"] == "env-key"
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    _, raw = generate_chart_bars_with_ai(_analysis(), "Oni", 10, "technical")

    assert raw["model"] == "openai/env-model"
    assert raw["api_base"] == "https://env.example.com/v1"
    assert raw["api_key_provided"] is True


def test_generate_chart_bars_with_ai_repairs_invalid_output(monkeypatch):
    responses = [
        {"bars": [{"bar": 1, "notes": "12x"}]},
        {"bars": [{"bar": 1, "notes": "1000100010001000"}]},
    ]
    captured_messages = []

    def fake_completion(model, messages, temperature):
        captured_messages.append(messages.copy())
        payload = responses.pop(0)
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, raw = generate_chart_bars_with_ai(
        _analysis(),
        "Oni",
        10,
        "technical",
        model="fake/model",
        max_repair_attempts=1,
    )

    assert bars == [ChartBar(index=0, notes="1000100010001000")]
    assert [attempt["status"] for attempt in raw["attempts"]] == ["invalid", "ok"]
    assert "Fix the output" in captured_messages[1][-1]["content"]


def test_generate_chart_bars_with_ai_accepts_variable_meter_note_lengths(monkeypatch):
    payload = {"bars": [{"bar": 1, "notes": "100010001000"}]}
    analysis = SongAnalysis(
        title="Song Title",
        artist=None,
        audio_file="song.mp3",
        ogg_file="song.ogg",
        bpm=120,
        offset=0,
        time_signature="3/4",
        bars=[
            BarFeature(
                index=0,
                start_time=0,
                end_time=1.5,
                energy=0.5,
                time_signature="3/4",
                grids_per_bar=12,
            )
        ],
    )

    def fake_completion(model, messages, temperature):
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, _ = generate_chart_bars_with_ai(analysis, "Oni", 10, "technical", model="fake/model")

    assert bars == [ChartBar(index=0, notes="100010001000", time_signature="3/4")]


def test_generate_chart_bars_with_ai_raises_with_attempt_log_after_failed_repairs(monkeypatch):
    payload = {"bars": [{"bar": 1, "notes": "12x"}]}

    def fake_completion(model, messages, temperature):
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    with pytest.raises(AiOutputRepairError) as error:
        generate_chart_bars_with_ai(
            _analysis(),
            "Oni",
            10,
            "technical",
            model="fake/model",
            max_repair_attempts=1,
        )

    assert "remained invalid" in str(error.value)
    assert [attempt["status"] for attempt in error.value.output["attempts"]] == [
        "invalid",
        "invalid",
    ]


def _analysis() -> SongAnalysis:
    return SongAnalysis(
        title="Song Title",
        artist=None,
        audio_file="song.mp3",
        ogg_file="song.ogg",
        bpm=120,
        offset=0,
        bars=[BarFeature(index=0, start_time=0, end_time=2, energy=0.5)],
    )
