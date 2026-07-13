import json

import pytest
from litellm import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from tja_ai_chartgen.ai.client import (
    AiOutputRepairError,
    AiProviderError,
    generate_chart_bars_with_ai,
    sanitize_ai_bars,
)
from tja_ai_chartgen.ai.prompts import build_chart_generation_payload, build_chart_generation_prompt
from tja_ai_chartgen.tja.model import (
    BarFeature,
    BarStructureFeature,
    ChartBar,
    PhraseFeature,
    ResolutionPlan,
    SongAnalysis,
    SpectralGridFeature,
)


def _event_bar(
    bar_number: int,
    notes: str,
    *,
    canonical_grids: int = 48,
    balloon_counts: list[int] | None = None,
) -> dict[str, object]:
    if not notes or canonical_grids % len(notes) != 0:
        raise ValueError("test notes must divide the canonical grid exactly")

    step = canonical_grids // len(notes)
    hits: list[list[object]] = []
    long_notes: list[dict[str, object]] = []
    balloons = iter(balloon_counts or [])
    active_long_start: tuple[int, str] | None = None

    for index, note in enumerate(notes):
        tick = index * step
        if note in "1234":
            hits.append([tick, note])
        elif note in "57":
            active_long_start = (tick, note)
        elif note == "8" and active_long_start is not None:
            start_tick, start_note = active_long_start
            long_note: dict[str, object] = {
                "start_tick": start_tick,
                "end_tick": tick,
                "kind": "balloon" if start_note == "7" else "drumroll",
            }
            if start_note == "7":
                long_note["balloon_count"] = next(balloons)
            long_notes.append(long_note)
            active_long_start = None

    return {"bar": bar_number, "hits": hits, "long_notes": long_notes}


def _event_payload(
    notes_by_bar: list[str],
    *,
    canonical_grids: int = 48,
) -> dict[str, list[dict[str, object]]]:
    return {
        "bars": [
            _event_bar(index + 1, notes, canonical_grids=canonical_grids)
            for index, notes in enumerate(notes_by_bar)
        ]
    }


def test_sanitize_ai_bars_rejects_bar_count_mismatch():
    with pytest.raises(ValueError, match="exactly 2 bars"):
        sanitize_ai_bars([ChartBar(index=10, notes="1000100010001000")], expected_count=2)


@pytest.mark.parametrize(
    ("notes", "message"),
    [
        ("12x0000000000000", "unsupported character"),
        ("12340123401234012340", "expected resolution 16"),
    ],
)
def test_sanitize_ai_bars_rejects_invalid_notes_instead_of_rewriting(notes, message):
    with pytest.raises(ValueError, match=message):
        sanitize_ai_bars([ChartBar(index=10, notes=notes)], expected_count=1)


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
    analysis = _analysis(energy=0.5)

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
    assert payload["schema"] == "tja-ai-chartgen-compact-v2"
    assert len(payload["reference_windows"]) == 3
    assert "reference_window_bar_columns" in payload["legend"]
    assert payload["bars"][0]["canonical_grids_per_bar"] == 48
    assert payload["bars"][0]["output_resolution"] == 16
    assert payload["bars"][0]["allowed_tick_step"] == 3
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


def test_build_chart_generation_payload_includes_compact_structure_plan():
    analysis = _analysis(bar_count=2).model_copy(
        update={
            "structure_feature_version": "structure-v1",
            "structure_confidence": 0.81,
            "bar_structures": [
                BarStructureFeature(
                    index=0,
                    energy_percentile=0.3,
                    phrase_id=0,
                    phrase_progress=0.0,
                    phrase_position="start",
                    transition_role="build_up",
                    section_id="section-1",
                    section="verse",
                    section_confidence=0.9,
                    fill_candidate_score=0.1,
                ),
                BarStructureFeature(
                    index=1,
                    energy_percentile=0.8,
                    energy_delta=0.5,
                    boundary_confidence=0.7,
                    phrase_id=0,
                    phrase_progress=1.0,
                    phrase_position="end",
                    transition_role="peak",
                    section_id="section-1",
                    section="verse",
                    section_confidence=0.9,
                    fill_candidate_score=0.75,
                ),
            ],
            "phrase_plan": [
                PhraseFeature(
                    phrase_id=0,
                    start_bar=0,
                    end_bar=1,
                    section_id="section-1",
                    section="verse",
                    mean_energy=0.55,
                    peak_energy=0.8,
                    energy_trend=0.5,
                    primary_role="build_up",
                    ending_boundary_confidence=0.7,
                    resolution=16,
                )
            ],
        }
    )

    payload = build_chart_generation_payload(analysis, "Oni", 10, "technical", "high")

    structure_columns = payload["legend"]["bar_structure_columns"]
    role_index = structure_columns.index("transition_role")
    fill_index = structure_columns.index("fill_candidate_score")
    assert payload["structure_feature_version"] == "structure-v1"
    assert payload["structure_confidence"] == 0.81
    assert payload["bar_structure"][0][role_index] == "build_up"
    assert payload["bar_structure"][1][fill_index] == 0.75
    assert payload["legend"]["phrase_columns"][-1] == "resolution"
    assert payload["phrase_plan"][0][-1] == 16


def test_build_chart_generation_payload_includes_compact_spectral_semantics():
    analysis = _analysis()
    spectral_bar = analysis.bars[0].model_copy(
        update={
            "low_onset_strength": 0.8,
            "mid_onset_strength": 0.3,
            "high_onset_strength": 0.6,
            "spectral_flux": 0.9,
            "brightness": 0.55,
            "harmonic_novelty": 0.7,
            "texture_novelty": 0.4,
            "percussive_ratio": 0.75,
            "spectral_grid_features": [
                SpectralGridFeature(
                    grid=12,
                    low_onset_strength=0.8,
                    high_onset_strength=0.2,
                    spectral_flux=0.9,
                )
            ],
        }
    )
    analysis = analysis.model_copy(
        update={
            "spectral_feature_version": "spectral-v1",
            "spectral_analysis_status": "complete",
            "bars": [spectral_bar],
        }
    )

    payload = build_chart_generation_payload(analysis, "Oni", 10, "technical")

    structure_columns = payload["legend"]["bar_structure_columns"]
    flux_index = structure_columns.index("spectral_flux")
    novelty_index = structure_columns.index("harmonic_novelty")
    assert payload["spectral_feature_version"] == "spectral-v1"
    assert payload["spectral_analysis_status"] == "complete"
    assert payload["bar_structure"][0][flux_index] == 0.9
    assert payload["bar_structure"][0][novelty_index] == 0.7
    assert payload["legend"]["spectral_grid_columns"] == [
        "grid",
        "low_onset",
        "mid_onset",
        "high_onset",
        "spectral_flux",
    ]
    assert payload["bars"][0]["spectral_events"] == [[12, 0.8, 0.0, 0.2, 0.9]]


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
    assert "Never output the legacy notes or balloon_counts fields" in prompt
    assert "For build_up roles, increase activity gradually across the whole phrase" in prompt
    assert "Do not create a fill merely because a bar number is divisible by 4 or 8" in prompt
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
    payload = _event_payload(["1000100010001000"])

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


def test_generate_chart_bars_with_ai_encodes_canonical_event_ticks(monkeypatch):
    payload = {
        "bars": [
            {
                "bar": 1,
                "hits": [[0, "1"], [12, "2"], [24, "1"], [36, "2"]],
                "long_notes": [],
            }
        ]
    }
    analysis = SongAnalysis(
        title="Song Title",
        audio_file="song.mp3",
        ogg_file="song.ogg",
        bpm=120,
        offset=0,
        resolution_plan=ResolutionPlan(
            canonical_grids_per_bar=48,
            base_resolution=16,
            bar_resolutions=[16],
        ),
        bars=[
            BarFeature(
                index=0,
                start_time=0,
                end_time=2,
                energy=0.2,
                grids_per_bar=48,
                onset_grids=[0, 12, 24, 36],
                beat_grids=[0, 12, 24, 36],
                downbeat_grid=0,
            )
        ],
    )

    monkeypatch.setattr(
        "tja_ai_chartgen.ai.client.completion",
        lambda **_kwargs: {"choices": [{"message": {"content": json.dumps(payload)}}]},
    )

    bars, _ = generate_chart_bars_with_ai(
        analysis,
        "Oni",
        10,
        "technical",
        model="fake/model",
    )

    assert bars == [ChartBar(index=0, notes="1000200010002000")]


def test_generate_chart_bars_with_ai_repairs_unrepresentable_event_tick(monkeypatch):
    responses = [
        {"bars": [{"bar": 1, "hits": [[1, "1"], [12, "2"]], "long_notes": []}]},
        {"bars": [{"bar": 1, "hits": [[0, "1"], [12, "2"]], "long_notes": []}]},
    ]
    analysis = SongAnalysis(
        title="Song Title",
        audio_file="song.mp3",
        ogg_file="song.ogg",
        bpm=120,
        offset=0,
        resolution_plan=ResolutionPlan(
            canonical_grids_per_bar=48,
            base_resolution=16,
            bar_resolutions=[16],
        ),
        bars=[
            BarFeature(
                index=0,
                start_time=0,
                end_time=2,
                energy=0.2,
                grids_per_bar=48,
                onset_grids=[0, 12, 24, 36],
            )
        ],
    )

    monkeypatch.setattr(
        "tja_ai_chartgen.ai.client.completion",
        lambda **_kwargs: {
            "choices": [{"message": {"content": json.dumps(responses.pop(0))}}]
        },
    )

    bars, output = generate_chart_bars_with_ai(
        analysis,
        "Oni",
        10,
        "technical",
        model="fake/model",
        max_repair_attempts=1,
    )

    assert bars == [ChartBar(index=0, notes="1000200000000000")]
    assert [attempt["status"] for attempt in output["attempts"]] == ["invalid", "ok"]
    assert "tick-not-representable" in output["attempts"][0]["issues"][0]


def test_generate_chart_bars_with_ai_passes_openai_compatible_connection_options(monkeypatch):
    payload = _event_payload(["1000100010001000"])
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

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

    serialized_raw = json.dumps(raw)
    assert raw["api_base"] == "https://llm.example.com/v1"
    assert raw["api_key_provided"] is True
    assert "test-key" not in serialized_raw
    assert "env-key" not in serialized_raw


def test_generate_chart_bars_with_ai_reads_openai_env_names(monkeypatch):
    payload = _event_payload(["1000100010001000"])
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


def test_generate_chart_bars_with_ai_passes_timeout_and_disables_litellm_retries(monkeypatch):
    payload = _event_payload(["1000100010001000"])
    captured_kwargs = []

    def fake_completion(**kwargs):
        captured_kwargs.append(kwargs)
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    _, raw = generate_chart_bars_with_ai(
        _analysis(),
        "Oni",
        10,
        "technical",
        request_timeout=12.5,
        max_transport_retries=0,
    )

    assert captured_kwargs[0]["timeout"] == 12.5
    assert captured_kwargs[0]["max_retries"] == 0
    assert raw["request_timeout"] == 12.5
    assert raw["max_transport_retries"] == 0


@pytest.mark.parametrize(
    "error_type",
    [Timeout, APIConnectionError, RateLimitError, ServiceUnavailableError, InternalServerError],
)
def test_generate_chart_bars_with_ai_retries_transient_transport_once_without_consuming_content_attempt(
    tmp_path, monkeypatch, error_type
):
    payload = _event_payload(["1000100010001000"])
    attempt_log_path = tmp_path / "ai_attempts.json"
    calls = 0

    def fake_completion(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error_type("temporary failure", model="fake/model", llm_provider="openai")
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    _, raw = generate_chart_bars_with_ai(
        _analysis(),
        "Oni",
        10,
        "technical",
        model="fake/model",
        max_repair_attempts=2,
        max_transport_retries=1,
        attempt_log_path=attempt_log_path,
    )

    assert calls == 2
    assert [attempt["status"] for attempt in raw["attempts"]] == ["ok"]
    assert [attempt["status"] for attempt in raw["transport_attempts"]] == ["error", "ok"]
    assert [attempt["content_attempt"] for attempt in raw["transport_attempts"]] == [1, 1]
    assert [attempt["transport_attempt"] for attempt in raw["transport_attempts"]] == [1, 2]
    assert raw["fallback_reason"] is None
    assert attempt_log_path.exists()


def test_generate_chart_bars_with_ai_stops_after_transport_retries_are_exhausted(
    tmp_path, monkeypatch
):
    attempt_log_path = tmp_path / "ai_attempts.json"
    calls = 0

    def fake_completion(**kwargs):
        nonlocal calls
        calls += 1
        raise Timeout("timed out", model="fake/model", llm_provider="openai")

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    with pytest.raises(AiProviderError) as error:
        generate_chart_bars_with_ai(
            _analysis(),
            "Oni",
            10,
            "technical",
            model="fake/model",
            max_repair_attempts=2,
            request_timeout=5,
            max_transport_retries=1,
            attempt_log_path=attempt_log_path,
        )

    assert calls == 2
    assert error.value.output["attempts"] == []
    assert len(error.value.output["transport_attempts"]) == 2
    assert error.value.output["fallback_reason"] == "transport_retries_exhausted"
    assert error.value.output["request_timeout"] == 5
    assert error.value.output["max_transport_retries"] == 1
    assert json.loads(attempt_log_path.read_text(encoding="utf-8"))["fallback_reason"] == (
        "transport_retries_exhausted"
    )


def test_generate_chart_bars_with_ai_does_not_retry_non_transient_provider_errors(monkeypatch):
    calls = 0

    def fake_completion(**kwargs):
        nonlocal calls
        calls += 1
        raise AuthenticationError("invalid key", model="fake/model", llm_provider="openai")

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    with pytest.raises(AiProviderError) as error:
        generate_chart_bars_with_ai(
            _analysis(),
            "Oni",
            10,
            "technical",
            model="fake/model",
            max_transport_retries=1,
        )

    assert calls == 1
    assert error.value.output["attempts"] == []
    assert len(error.value.output["transport_attempts"]) == 1
    assert error.value.output["fallback_reason"] == "provider_error"
    assert error.value.output["transport_attempts"][0]["error_type"] == "AuthenticationError"


@pytest.mark.parametrize("request_timeout", [0, 0.999, 600.001, 601])
def test_generate_chart_bars_with_ai_rejects_request_timeout_outside_bounds(
    monkeypatch, request_timeout
):
    completion_called = False

    def fake_completion(**kwargs):
        nonlocal completion_called
        completion_called = True

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    with pytest.raises(ValueError, match="between 1 and 600"):
        generate_chart_bars_with_ai(
            _analysis(),
            "Oni",
            10,
            "technical",
            request_timeout=request_timeout,
        )

    assert completion_called is False


@pytest.mark.parametrize("max_transport_retries", [-1, 2])
def test_generate_chart_bars_with_ai_rejects_transport_retry_count_outside_bounds(
    monkeypatch, max_transport_retries
):
    completion_called = False

    def fake_completion(**kwargs):
        nonlocal completion_called
        completion_called = True

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    with pytest.raises(ValueError, match="0 or 1"):
        generate_chart_bars_with_ai(
            _analysis(),
            "Oni",
            10,
            "technical",
            max_transport_retries=max_transport_retries,
        )

    assert completion_called is False


def test_generate_chart_bars_with_ai_redacts_api_key_from_provider_error_and_sidecar(
    tmp_path, monkeypatch
):
    api_key = "secret-provider-key"
    attempt_log_path = tmp_path / "ai_attempts.json"

    def fake_completion(**kwargs):
        raise AuthenticationError(
            f"invalid key: {api_key}",
            model="fake/model",
            llm_provider="openai",
        )

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    with pytest.raises(AiProviderError) as error:
        generate_chart_bars_with_ai(
            _analysis(),
            "Oni",
            10,
            "technical",
            model="fake/model",
            api_key=api_key,
            attempt_log_path=attempt_log_path,
        )

    serialized_output = json.dumps(error.value.output)
    logged = attempt_log_path.read_text(encoding="utf-8")
    assert api_key not in str(error.value)
    assert api_key not in serialized_output
    assert api_key not in logged
    assert "[REDACTED]" in serialized_output


def test_generate_chart_bars_with_ai_repairs_legacy_notes_schema(monkeypatch):
    responses = [
        {"bars": [{"bar": 1, "notes": "1000100010001000"}]},
        _event_payload(["1000100010001000"]),
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
    assert "legacy notes schema" in raw["attempts"][0]["issues"][0]
    assert "Fix the output" in captured_messages[1][-1]["content"]


def test_generate_chart_bars_with_ai_writes_invalid_attempt_log(tmp_path, monkeypatch):
    invalid_payload = {
        "bars": [
            {"bar": 1, "hits": [["0", "1"]], "long_notes": []},
            {"bar": 2, "hits": [[0, "9"]], "long_notes": []},
        ]
    }
    responses = [
        invalid_payload,
        _event_payload(["1000100010001000", "1000100010001000"]),
    ]
    attempt_log_path = tmp_path / "ai_attempts.json"

    def fake_completion(**kwargs):
        payload = responses.pop(0)
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, raw = generate_chart_bars_with_ai(
        _analysis(bar_count=2),
        "Oni",
        10,
        "technical",
        model="fake/model",
        max_repair_attempts=1,
        attempt_log_path=attempt_log_path,
    )

    logged = json.loads(attempt_log_path.read_text(encoding="utf-8"))
    assert bars == [
        ChartBar(index=0, notes="1000100010001000"),
        ChartBar(index=1, notes="1000100010001000"),
    ]
    assert [attempt["status"] for attempt in logged["attempts"]] == ["invalid", "ok"]
    assert logged["attempts"][0]["content"] == json.dumps(invalid_payload)
    assert any("must be an integer" in issue for issue in logged["attempts"][0]["issues"])
    assert any("unsupported-hit-note" in issue for issue in logged["attempts"][0]["issues"])
    assert logged["final"] == raw["final"]



def test_generate_chart_bars_with_ai_accepts_special_notes_with_balloon_counts(monkeypatch):
    payload = {
        "bars": [
            _event_bar(
                1,
                "7000000080000000",
                balloon_counts=[8],
            )
        ]
    }

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
    sparse_payload = _event_payload(["1000000000000000"] * 8)
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
    dense_payload = _event_payload(dense_notes)
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


def test_generate_chart_bars_with_ai_repairs_sparse_single_bar_output(monkeypatch):
    responses = [
        _event_payload(["0000000000000000"]),
        _event_payload(["1000100010001000"]),
    ]
    captured_messages = []

    def fake_completion(**kwargs):
        captured_messages.append(kwargs["messages"].copy())
        return {"choices": [{"message": {"content": json.dumps(responses.pop(0))}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, raw = generate_chart_bars_with_ai(
        _analysis(),
        "Oni",
        10,
        "technical",
        model="fake/model",
        max_repair_attempts=1,
    )

    assert [attempt["status"] for attempt in raw["attempts"]] == ["invalid", "ok"]
    assert bars == [ChartBar(index=0, notes="1000100010001000")]
    assert "too sparse for normal density hint" in captured_messages[1][-1]["content"]


def test_generate_chart_bars_with_ai_repairs_dense_four_bar_output(monkeypatch):
    dense_payload = _event_payload(["1111111111111111"] * 4)
    fixed_payload = _event_payload(["1000100010001000"] * 4)
    responses = [dense_payload, fixed_payload]
    captured_messages = []

    def fake_completion(**kwargs):
        captured_messages.append(kwargs["messages"].copy())
        return {"choices": [{"message": {"content": json.dumps(responses.pop(0))}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, raw = generate_chart_bars_with_ai(
        _analysis(bar_count=4),
        "Oni",
        10,
        "technical",
        model="fake/model",
        max_repair_attempts=1,
    )

    assert [attempt["status"] for attempt in raw["attempts"]] == ["invalid", "ok"]
    assert [bar.notes for bar in bars] == ["1000100010001000"] * 4
    assert "too dense for normal density hint" in captured_messages[1][-1]["content"]


def test_generate_chart_bars_with_ai_accepts_valid_seven_bar_output(monkeypatch):
    payload = _event_payload(["1000100000000000"] * 7)
    call_count = 0

    def fake_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, raw = generate_chart_bars_with_ai(
        _analysis(bar_count=7),
        "Oni",
        10,
        "technical",
        model="fake/model",
        max_repair_attempts=1,
    )

    assert call_count == 1
    assert raw["attempts"][0]["status"] == "ok"
    assert [bar.notes for bar in bars] == ["1000100000000000"] * 7


def test_generate_chart_bars_with_ai_allows_empty_musical_rest_in_high_density(monkeypatch):
    payload = _event_payload(
        [
            "1022101210201220",
            "1212102210121020",
            "0000000000000000",
            "1022121010221010",
            "1210201210221020",
            "1022101212101022",
            "1212102010221012",
            "1022121010201220",
        ]
    )

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
    don_payload = _event_payload(["1010101010101010"] * 8)
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
    mixed_payload = _event_payload(mixed_notes)
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
    noisy_payload = _event_payload(
        [
            "1000100010001000",
            "1010101010101011",
            "1000100010001000",
        ]
    )
    fixed_payload = _event_payload(
        [
            "0000000000000000",
            "1010101010101011",
            "0000000000000000",
        ]
    )
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
    payload = _event_payload(["100010001000"], canonical_grids=36)
    analysis = SongAnalysis(
        title="Song Title",
        artist=None,
        audio_file="song.mp3",
        ogg_file="song.ogg",
        bpm=120,
        offset=0,
        time_signature="3/4",
        resolution_plan=ResolutionPlan(
            canonical_grids_per_bar=36,
            base_resolution=12,
            bar_resolutions=[12],
        ),
        bars=[
            BarFeature(
                index=0,
                start_time=0,
                end_time=1.5,
                energy=0.2,
                time_signature="3/4",
                grids_per_bar=36,
            )
        ],
    )

    def fake_completion(**kwargs):
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr("tja_ai_chartgen.ai.client.completion", fake_completion)

    bars, _ = generate_chart_bars_with_ai(analysis, "Oni", 10, "technical", model="fake/model")

    assert bars == [ChartBar(index=0, notes="100010001000", time_signature="3/4")]


def test_generate_chart_bars_with_ai_raises_with_attempt_log_after_failed_repairs(monkeypatch):
    payload = {"bars": [{"bar": 1, "hits": [[1, "1"]], "long_notes": []}]}

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
    assert error.value.output["fallback_reason"] == "invalid_content_retries_exhausted"
    assert [attempt["status"] for attempt in error.value.output["transport_attempts"]] == [
        "ok",
        "ok",
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
                    grids_per_bar=48,
                    onset_grids=[],
                    beat_grids=[0, 12, 24, 36],
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
                    grids_per_bar=48,
                    onset_grids=[0, 6, 12, 18, 24, 30, 36, 42],
                    beat_grids=[0, 12, 24, 36],
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
        resolution_plan=ResolutionPlan(
            canonical_grids_per_bar=48,
            base_resolution=16,
            bar_resolutions=[16] * len(bars),
        ),
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
        resolution_plan=ResolutionPlan(
            canonical_grids_per_bar=48,
            base_resolution=16,
            bar_resolutions=[16, 16, 16],
        ),
        bars=[
            BarFeature(
                index=0,
                start_time=0,
                end_time=2,
                energy=0.068,
                grids_per_bar=48,
                onset_grids=[0, 33, 36],
                rms_dbfs=-59.7,
                peak_rms_dbfs=-46.7,
                relative_rms_db=-53.5,
                sustained_activity_ratio=0.098,
                phrase_position="phrase_start",
                section="intro",
            ),
            BarFeature(
                index=1,
                start_time=2,
                end_time=4,
                energy=0.5,
                grids_per_bar=48,
                onset_grids=[0, 12, 24, 36],
                beat_grids=[0, 12, 24, 36],
                downbeat_grid=0,
                phrase_position="phrase_middle",
                section="verse",
            ),
            BarFeature(
                index=2,
                start_time=4,
                end_time=6,
                energy=0,
                grids_per_bar=48,
                phrase_position="song_end",
                section="outro",
            ),
        ],
    )


def _analysis(bar_count: int = 1, energy: float = 0.2) -> SongAnalysis:
    return SongAnalysis(
        title="Song Title",
        artist=None,
        audio_file="song.mp3",
        ogg_file="song.ogg",
        bpm=120,
        offset=0,
        resolution_plan=ResolutionPlan(
            canonical_grids_per_bar=48,
            base_resolution=16,
            bar_resolutions=[16] * bar_count,
        ),
        bars=[
            BarFeature(
                index=index,
                start_time=index * 2,
                end_time=(index + 1) * 2,
                energy=energy,
                grids_per_bar=48,
                onset_grids=[0, 12, 24, 36],
                beat_grids=[0, 12, 24, 36],
                downbeat_grid=0,
                phrase_position="phrase_start" if index % 4 == 0 else "phrase_middle",
                section="verse",
            )
            for index in range(bar_count)
        ],
    )
