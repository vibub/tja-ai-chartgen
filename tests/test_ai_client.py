import json

import pytest

from tja_ai_chartgen.ai.client import (
    AiOutputRepairError,
    generate_chart_bars_with_ai,
    sanitize_ai_bars,
)
from tja_ai_chartgen.ai.prompts import build_chart_generation_payload, build_chart_generation_prompt
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


def test_sanitize_ai_bars_forces_expected_edge_silence_to_empty():
    bars = [
        ChartBar(index=0, notes="1000100010001000"),
        ChartBar(index=1, notes="1010101010101011"),
        ChartBar(index=2, notes="1000100010001000"),
    ]

    sanitized = sanitize_ai_bars(
        bars,
        expected_count=3,
        expected_bars=_analysis_with_edge_silence().bars,
    )

    assert [bar.notes for bar in sanitized] == [
        "0000000000000000",
        "1010101010101011",
        "0000000000000000",
    ]


def test_build_chart_generation_payload_includes_density():
    analysis = _analysis()

    payload = build_chart_generation_payload(analysis, "Oni", 10, "technical", "high")

    assert payload["density"] == "high"
    assert payload["density_target"]["average_hits_per_16_grid_bar"] == "8-11"
    assert payload["forced_silent_bars"] == []
    hint_columns = payload["legend"]["bar_density_hint_columns"]
    kind_index = hint_columns.index("kind")
    target_index = hint_columns.index("target_hits")
    assert payload["bar_density_hints"][0][kind_index] == "dense"
    assert payload["bar_density_hints"][0][target_index] == 10
    assert payload["density_policy"]["quality_density"] == "high"
    assert payload["density_policy"]["quality_average_min_per_16_grid_bar"] == 6.5
    assert "note_color_target" not in payload
    assert payload["style"] == "technical"
    assert payload["schema"] == "tja-ai-chartgen-compact-v1"
    assert payload["bars"][0]["grids_per_bar"] == 16
    assert "grid_features" in payload["bars"][0]
    assert payload["legend"]["grid_feature_columns"] == [
        "grid",
        "onset",
        "accent",
        "beat",
        "downbeat",
        "strength",
        "activity",
    ]


def test_build_chart_generation_payload_can_include_static_reference_prompt():
    analysis = _analysis()

    payload = build_chart_generation_payload(
        analysis,
        "Oni",
        10,
        "technical",
        reference_examples_prompt="static reference prompt",
    )

    assert payload["reference_examples_prompt"] == "static reference prompt"


def test_build_chart_generation_prompt_constrains_big_notes_for_playability():
    prompt = build_chart_generation_prompt(
        _analysis(),
        "Oni",
        10,
        "performance",
        density="high",
        reference_examples_prompt="static reference prompt",
    )

    assert "Grid 0 is the barline and primary downbeat candidate" in prompt
    assert "prefer starting the bar with a 1/2 note on grid 0" in prompt
    assert "Big notes 3/4 require both hands hitting together" in prompt
    assert "Do not place big notes 3/4 inside dense streams" in prompt
    assert "3 or more consecutive playable hits" in prompt
    assert "without overusing big notes" in prompt
    assert "Density and difficulty targets" in prompt
    assert "bar_density_hints" in prompt
    assert "forced_silent_bars" in prompt
    assert "bar_density_hints" in prompt
    assert "note_color_target" not in prompt
    assert "Do not use 1 as the default" in prompt
    assert "do not force a fixed ratio" in prompt
    assert "1010101010101010" in prompt
    assert "target_hits is more important than merely satisfying min_hits" in prompt
    assert "density_policy.quality_average_min_per_16_grid_bar" in prompt


def test_generate_chart_bars_with_ai_parses_litellm_dict_response(monkeypatch):
    payload = {"bars": [{"bar": 1, "notes": "1000100010001000"}]}

    def fake_completion(**kwargs):
        assert kwargs["model"] == "fake/model"
        assert kwargs["messages"][0]["role"] == "user"
        assert "Reference chart examples" in kwargs["messages"][0]["content"]
        assert kwargs["temperature"] == 0.7
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

    def fake_completion(**kwargs):
        captured_messages.append(kwargs["messages"].copy())
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


def test_generate_chart_bars_with_ai_accepts_special_notes_with_balloon_counts(monkeypatch):
    payload = {"bars": [{"bar": 1, "notes": "7000000080000000", "balloon_counts": [8]}]}

    def fake_completion(**kwargs):
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, _ = generate_chart_bars_with_ai(
        _analysis(),
        "Oni",
        10,
        "technical",
        model="fake/model",
        special_notes=True,
    )

    assert bars == [ChartBar(index=0, notes="7000000080000000", balloon_counts=[8])]


def test_generate_chart_bars_with_ai_repairs_sparse_high_density_output(monkeypatch):
    sparse_payload = {
        "bars": [{"bar": index + 1, "notes": "1000000000000000"} for index in range(8)]
    }
    dense_notes = [
        "1022101210201220",
        "1212102210121020",
        "1022121010221010",
        "1210201210221020",
        "1022101212101022",
        "1212102010221012",
        "1022121010201220",
        "1210202210121020",
    ]
    dense_payload = {
        "bars": [{"bar": index + 1, "notes": notes} for index, notes in enumerate(dense_notes)]
    }
    responses = [sparse_payload, dense_payload]
    captured_messages = []

    def fake_completion(**kwargs):
        captured_messages.append(kwargs["messages"].copy())
        return {"choices": [{"message": {"content": json.dumps(responses.pop(0))}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, raw = generate_chart_bars_with_ai(
        _analysis(bar_count=8, energy=0.5),
        "Oni",
        10,
        "technical",
        density="max",
        model="fake/model",
        max_repair_attempts=1,
    )

    assert [attempt["status"] for attempt in raw["attempts"]] == ["invalid", "ok"]
    assert [bar.notes for bar in bars] == dense_notes
    assert "chart quality is too sparse" in captured_messages[1][-1]["content"]
    assert "toward target_hits" in captured_messages[1][-1]["content"]


def test_generate_chart_bars_with_ai_allows_empty_musical_rest_in_high_density(monkeypatch):
    payload = {
        "bars": [
            {"bar": 1, "notes": "1022101210201220"},
            {"bar": 2, "notes": "1212102210121020"},
            {"bar": 3, "notes": "0000000000000000"},
            {"bar": 4, "notes": "1022121010221010"},
            {"bar": 5, "notes": "1210201210221020"},
            {"bar": 6, "notes": "1022101212101022"},
            {"bar": 7, "notes": "1212102010221012"},
            {"bar": 8, "notes": "1022121010201220"},
        ]
    }

    def fake_completion(**kwargs):
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, raw = generate_chart_bars_with_ai(
        _analysis_with_middle_rest(),
        "Oni",
        10,
        "technical",
        density="max",
        model="fake/model",
        max_repair_attempts=0,
    )

    assert raw["attempts"][0]["status"] == "ok"
    assert bars[2].notes == "0000000000000000"


def test_generate_chart_bars_with_ai_repairs_all_don_output(monkeypatch):
    don_payload = {
        "bars": [
            {"bar": index + 1, "notes": "1010101010101010"}
            for index in range(8)
        ]
    }
    mixed_notes = [
        "1020102010201020",
        "1012101210121022",
        "1022101210221012",
        "1210102012101020",
        "1020102210201012",
        "1012102010121020",
        "1022101210201220",
        "1210102012102012",
    ]
    mixed_payload = {
        "bars": [{"bar": index + 1, "notes": notes} for index, notes in enumerate(mixed_notes)]
    }
    responses = [don_payload, mixed_payload]
    captured_messages = []

    def fake_completion(**kwargs):
        captured_messages.append(kwargs["messages"].copy())
        return {"choices": [{"message": {"content": json.dumps(responses.pop(0))}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, raw = generate_chart_bars_with_ai(
        _analysis(bar_count=8, energy=0.5),
        "Oni",
        8,
        "performance",
        density="low",
        model="fake/model",
        max_repair_attempts=1,
    )

    assert [attempt["status"] for attempt in raw["attempts"]] == ["invalid", "ok"]
    assert [bar.notes for bar in bars] == mixed_notes
    assert "nearly all don notes" in captured_messages[1][-1]["content"]


def test_generate_chart_bars_with_ai_repairs_notes_in_edge_silence(monkeypatch):
    noisy_payload = {
        "bars": [
            {"bar": 1, "notes": "1000100010001000"},
            {"bar": 2, "notes": "1010101010101011"},
            {"bar": 3, "notes": "1000100010001000"},
        ]
    }
    fixed_payload = {
        "bars": [
            {"bar": 1, "notes": "0000000000000000"},
            {"bar": 2, "notes": "1010101010101011"},
            {"bar": 3, "notes": "0000000000000000"},
        ]
    }
    responses = [noisy_payload, fixed_payload]
    captured_messages = []

    def fake_completion(**kwargs):
        captured_messages.append(kwargs["messages"].copy())
        return {"choices": [{"message": {"content": json.dumps(responses.pop(0))}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, raw = generate_chart_bars_with_ai(
        _analysis_with_edge_silence(),
        "Oni",
        10,
        "technical",
        density="max",
        model="fake/model",
        max_repair_attempts=1,
    )

    assert [attempt["status"] for attempt in raw["attempts"]] == ["invalid", "ok"]
    assert [bar.notes for bar in bars] == [
        "0000000000000000",
        "1010101010101011",
        "0000000000000000",
    ]
    assert "song-start/song-end silence" in captured_messages[1][-1]["content"]


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

    def fake_completion(**kwargs):
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, _ = generate_chart_bars_with_ai(analysis, "Oni", 10, "technical", model="fake/model")

    assert bars == [ChartBar(index=0, notes="100010001000", time_signature="3/4")]


def test_generate_chart_bars_with_ai_raises_with_attempt_log_after_failed_repairs(monkeypatch):
    payload = {"bars": [{"bar": 1, "notes": "12x"}]}

    def fake_completion(**kwargs):
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


def _analysis_with_middle_rest() -> SongAnalysis:
    bars = []
    for index in range(8):
        if index == 2:
            bars.append(
                BarFeature(
                    index=index,
                    start_time=index * 2,
                    end_time=(index + 1) * 2,
                    energy=0.01,
                    onset_16=[],
                    beat_grids=[0, 4, 8, 12],
                    downbeat_grid=0,
                    phrase_position="phrase_middle",
                    section="break",
                )
            )
        else:
            bars.append(
                BarFeature(
                    index=index,
                    start_time=index * 2,
                    end_time=(index + 1) * 2,
                    energy=0.6,
                    onset_16=[0, 2, 4, 6, 8, 10, 12, 14],
                    beat_grids=[0, 4, 8, 12],
                    downbeat_grid=0,
                    phrase_position="phrase_start" if index % 4 == 0 else "phrase_middle",
                    section="verse",
                )
            )
    return SongAnalysis(
        title="Song Title",
        artist=None,
        audio_file="song.mp3",
        ogg_file="song.ogg",
        bpm=120,
        offset=0,
        bars=bars,
    )


def _analysis_with_edge_silence() -> SongAnalysis:
    return SongAnalysis(
        title="Song Title",
        artist=None,
        audio_file="song.mp3",
        ogg_file="song.ogg",
        bpm=120,
        offset=0,
        bars=[
            BarFeature(
                index=0,
                start_time=0,
                end_time=2,
                energy=0,
                phrase_position="phrase_start",
                section="intro",
            ),
            BarFeature(
                index=1,
                start_time=2,
                end_time=4,
                energy=0.5,
                onset_16=[0, 4, 8, 12],
                beat_grids=[0, 4, 8, 12],
                downbeat_grid=0,
                phrase_position="phrase_middle",
                section="verse",
            ),
            BarFeature(
                index=2,
                start_time=4,
                end_time=6,
                energy=0,
                phrase_position="song_end",
                section="outro",
            ),
        ],
    )


def _analysis(bar_count: int = 1, energy: float = 0.5) -> SongAnalysis:
    return SongAnalysis(
        title="Song Title",
        artist=None,
        audio_file="song.mp3",
        ogg_file="song.ogg",
        bpm=120,
        offset=0,
        bars=[
            BarFeature(
                index=index,
                start_time=index * 2,
                end_time=(index + 1) * 2,
                energy=energy,
                onset_16=[0, 4, 8, 12],
                beat_grids=[0, 4, 8, 12],
                downbeat_grid=0,
                phrase_position="phrase_start" if index % 4 == 0 else "phrase_middle",
                section="verse",
            )
            for index in range(bar_count)
        ],
    )
