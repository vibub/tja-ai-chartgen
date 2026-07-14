from pathlib import Path
import shutil

import pytest

from tja_ai_chartgen.audio.analyze import analyze_audio, apply_analysis_overrides
from tja_ai_chartgen.audio.convert import convert_to_ogg
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.features.density import build_density_hints
from tja_ai_chartgen.features.resolution import build_resolution_plan
from tja_ai_chartgen.features.structure import analyze_song_structure
from tja_ai_chartgen.generation import (
    build_analysis_notices,
    build_song_analysis,
    generate_chart_bars,
)
from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars
from tja_ai_chartgen.tja.model import (
    ChartBar,
    ChartMetadata,
    InstrumentBarFeature,
    SongAnalysis,
    TjaChart,
)
from tja_ai_chartgen.tja.quality import build_quality_report, playable_hit_count
from tja_ai_chartgen.tja.writer import TJA_FILE_ENCODING, render_tja, write_tja_text


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"


@pytest.mark.parametrize(
    ("fixture_name", "expected_duration", "expected_offset"),
    [
        ("click_4_4.wav", 8.25, 0.25),
        ("click_4_4_leadin.wav", 9.25, 0.25),
    ],
)
def test_real_audio_pipeline(
    tmp_path: Path,
    fixture_name: str,
    expected_duration: float,
    expected_offset: float,
):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the real audio pipeline integration test")

    fixture_path = FIXTURE_DIR / fixture_name
    assert fixture_path.is_file(), f"Missing golden audio fixture: {fixture_path}"

    ogg_path = convert_to_ogg(fixture_path, tmp_path / f"{fixture_path.stem}.ogg")

    assert ogg_path.is_file()
    assert ogg_path.stat().st_size > 0

    raw = analyze_audio(ogg_path)

    assert raw.bpm == pytest.approx(120.0, abs=1.0)
    assert raw.offset == pytest.approx(expected_offset, abs=0.08)
    assert raw.duration == pytest.approx(expected_duration, abs=0.08)
    assert raw.sample_rate == 22050
    assert raw.rms_envelope
    assert len(raw.rms_envelope) == len(raw.activity_envelope)
    assert len(raw.onset_times) >= 12
    assert raw.beat_times
    assert raw.analyzer == "librosa+onset-grid"
    assert raw.tempo_analysis is not None
    assert raw.tempo_analysis.accepted is True
    assert raw.tempo_analysis.reason == "accepted"
    assert raw.tempo_analysis.selected_source == "onset-grid"
    assert raw.tempo_analysis.normalized_support >= 0.85
    assert raw.tempo_analysis.onset_count >= 12
    assert raw.tempo_analysis.time_coverage >= 0.75
    assert raw.spectral.status == "complete"
    assert raw.spectral.feature_version == "spectral-v1"
    assert raw.spectral.frame_count == len(raw.spectral.spectral_flux_envelope)
    assert any(value > 0.0 for value in raw.spectral.spectral_flux_envelope)
    assert raw.instruments.status == "unavailable"
    assert raw.instruments.feature_version is None

    features = build_bar_features(raw)

    assert features
    assert all(bar.time_signature == "4/4" for bar in features)
    assert all(bar.grids_per_bar == 48 for bar in features)
    assert any(bar.onset_grids for bar in features)
    assert any(bar.spectral_flux > 0.0 for bar in features)
    assert any(bar.spectral_grid_features for bar in features)

    analysis = SongAnalysis(
        spectral_feature_version=raw.spectral.feature_version,
        spectral_analysis_status=raw.spectral.status,
        spectral_analysis_reason=raw.spectral.reason,
        title="Golden Click Track",
        audio_file=str(fixture_path),
        ogg_file=str(ogg_path),
        bpm=raw.bpm,
        offset=raw.offset,
        analyzer=raw.analyzer,
        tempo_analysis=raw.tempo_analysis,
        bars=features,
    )
    serialized_analysis = analysis.model_dump(mode="json")

    assert serialized_analysis["analyzer"] == "librosa+onset-grid"
    assert serialized_analysis["spectral_feature_version"] == "spectral-v1"
    assert serialized_analysis["spectral_analysis_status"] == "complete"
    assert serialized_analysis["instrument_feature_version"] is None
    assert serialized_analysis["instrument_analysis_status"] == "unavailable"
    assert serialized_analysis["tempo_analysis"]["accepted"] is True
    assert serialized_analysis["tempo_analysis"]["fallback_source"] == "librosa"
    assert serialized_analysis["bars"][0]["rms_dbfs"] is not None
    assert serialized_analysis["bars"][0]["sustained_activity_ratio"] is not None

    legacy_analysis = serialized_analysis.copy()
    legacy_analysis.pop("analyzer")
    legacy_analysis.pop("tempo_analysis")
    legacy_analysis.pop("spectral_feature_version")
    legacy_analysis.pop("spectral_analysis_status")
    legacy_analysis.pop("spectral_analysis_reason")
    legacy_analysis.pop("instrument_feature_version")
    legacy_analysis.pop("instrument_analysis_status")
    legacy_analysis.pop("instrument_analysis_reason")
    legacy_analysis.pop("instrument_demucs_model")
    legacy_analysis.pop("instrument_classifier_model")
    legacy_analysis.pop("instrument_analysis_device")
    for bar in legacy_analysis["bars"]:
        bar["onset_16"] = bar.pop("onset_grids")
        bar["accent_16"] = bar.pop("accent_grids")
        bar["activity_16"] = bar.pop("activity_grids")
        bar.pop("rms_dbfs")
        bar.pop("peak_rms_dbfs")
        bar.pop("relative_rms_db")
        bar.pop("sustained_activity_ratio")
        bar.pop("spectral_grid_features")
        bar.pop("low_onset_strength")
        bar.pop("mid_onset_strength")
        bar.pop("high_onset_strength")
        bar.pop("spectral_flux")
        bar.pop("brightness")
        bar.pop("harmonic_novelty")
        bar.pop("texture_novelty")
        bar.pop("percussive_ratio")
        bar.pop("instrument_grid_features")
        bar.pop("instrument")
    restored_legacy = SongAnalysis.model_validate(legacy_analysis)
    assert restored_legacy.analyzer == "unknown"
    assert restored_legacy.tempo_analysis is None
    assert restored_legacy.spectral_feature_version is None
    assert restored_legacy.spectral_analysis_status == "unavailable"
    assert restored_legacy.instrument_feature_version is None
    assert restored_legacy.instrument_analysis_status == "unavailable"
    assert restored_legacy.instrument_analysis_reason is None
    assert restored_legacy.bars[0].rms_dbfs is None
    assert restored_legacy.bars[0].spectral_grid_features == []
    assert restored_legacy.bars[0].spectral_flux == 0.0
    assert restored_legacy.bars[0].instrument_grid_features == []
    assert restored_legacy.bars[0].instrument == InstrumentBarFeature()

    chart = TjaChart(
        metadata=ChartMetadata(
            title="Golden Click Track",
            wave=ogg_path.name,
            bpm=raw.bpm,
            offset=raw.offset,
        ),
        bars=[
            ChartBar(
                index=bar.index,
                notes="1000100010001000",
                time_signature=bar.time_signature,
            )
            for bar in features
        ],
    )
    tja_text = render_tja(chart)
    tja_offset = float(
        next(line for line in tja_text.splitlines() if line.startswith("OFFSET:")).partition(":")[2]
    )

    assert tja_offset == pytest.approx(-raw.offset, abs=1e-6)

    tja_path = write_tja_text(tmp_path / f"{fixture_path.stem}.tja", tja_text)
    written_text = tja_path.read_text(encoding=TJA_FILE_ENCODING)

    assert "TITLE:Golden Click Track" in written_text
    assert f"WAVE:{ogg_path.name}" in written_text
    assert "#START" in written_text
    assert "#END" in written_text


def test_missing_instrument_models_fall_back_without_blocking_real_audio_generation(
    tmp_path: Path,
):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the real audio pipeline integration test")

    fixture_path = FIXTURE_DIR / "click_4_4.wav"
    analysis = build_song_analysis(
        input_audio=fixture_path,
        ogg_path=tmp_path / "click_4_4.ogg",
        title="Instrument Fallback",
        max_bars=2,
        use_instrument_analysis=True,
        instrument_model_dir=tmp_path / "missing-models",
    )

    assert analysis.analysis_schema_version == 5
    assert analysis.instrument_feature_version == "instrument-v1"
    assert analysis.instrument_analysis_status == "fallback"
    assert analysis.instrument_analysis_reason == "missing-model:manifest"
    assert analysis.bars
    assert all(bar.instrument == InstrumentBarFeature() for bar in analysis.bars)
    assert all(bar.instrument_grid_features == [] for bar in analysis.bars)

    notices = build_analysis_notices(
        analysis,
        requested_instrument_analysis=True,
    )
    assert any(notice.code == "instrument-models-missing" for notice in notices)

    generated = generate_chart_bars(
        analysis=analysis,
        selected_bars=analysis.bars,
        course="Oni",
        level=10,
        style="hybrid",
        density="auto",
        special_notes=False,
        use_ai=False,
    )
    chart = TjaChart(
        metadata=ChartMetadata(
            title=analysis.title,
            wave=Path(analysis.ogg_file).name,
            bpm=analysis.bpm,
            offset=analysis.offset,
            course="Oni",
            level=10,
        ),
        bars=generated.chart_bars,
    )

    assert generated.used_fallback is True
    assert generated.chart_bars
    assert "#END" in render_tja(chart)


@pytest.mark.parametrize(
    ("fixture_name", "expected_bpm", "expected_offset"),
    [
        ("sparse_120.wav", 120.0, 0.5),
        ("dense_180.wav", 180.0, 0.5),
    ],
)
def test_quality_audio_fixtures_support_repeatable_course_evaluation(
    tmp_path: Path,
    fixture_name: str,
    expected_bpm: float,
    expected_offset: float,
):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the real audio pipeline integration test")

    fixture_path = FIXTURE_DIR / fixture_name
    assert fixture_path.is_file(), f"Missing quality audio fixture: {fixture_path}"
    ogg_path = convert_to_ogg(fixture_path, tmp_path / f"{fixture_path.stem}.ogg")
    raw = analyze_audio(ogg_path)
    features = build_bar_features(raw)

    assert raw.bpm == pytest.approx(expected_bpm, abs=6.0)
    assert raw.offset == pytest.approx(expected_offset, abs=0.08)
    assert features

    course_results = []
    for course, level in [("Easy", 3), ("Normal", 5), ("Hard", 7), ("Oni", 10)]:
        first_bars = generate_fallback_chart_bars(
            features,
            density="auto",
            course=course,
            level=level,
        )
        second_bars = generate_fallback_chart_bars(
            features,
            density="auto",
            course=course,
            level=level,
        )
        first_report = build_quality_report(first_bars, features)
        second_report = build_quality_report(second_bars, features)

        assert first_bars == second_bars
        assert first_report == second_report
        assert first_report.silent_bar_note_count == 0
        assert first_report.accent_coverage_rate >= 0.5

        chart = TjaChart(
            metadata=ChartMetadata(
                title=f"Quality Fixture {course}",
                wave=ogg_path.name,
                bpm=raw.bpm,
                offset=raw.offset,
                course=course,
                level=level,
            ),
            bars=first_bars,
        )
        assert "#END" in render_tja(chart)
        course_results.append((first_bars, first_report))

    note_counts = [result.playable_note_count for _, result in course_results]
    assert note_counts == sorted(note_counts)
    assert all(lower < higher for lower, higher in zip(note_counts, note_counts[1:]))

    if fixture_name == "sparse_120.wav":
        oni_bars, _ = course_results[-1]
        for chart_bar, feature_bar in zip(oni_bars, features, strict=True):
            if len(feature_bar.onset_grids) <= 2:
                assert playable_hit_count(chart_bar.notes) <= 4
    else:
        hard_report = course_results[-2][1]
        oni_report = course_results[-1][1]
        assert oni_report.active_average_notes_per_second > (
            hard_report.active_average_notes_per_second
        )
        assert oni_report.longest_note_stream_count > hard_report.longest_note_stream_count


@pytest.mark.parametrize(
    ("fixture_name", "pattern", "expected_resolution"),
    [
        ("straight_120.wav", "straight", 16),
        ("triplet_120.wav", "triplet", 24),
        ("mixed_120.wav", "mixed", 48),
    ],
)
def test_resolution_audio_fixtures_select_stable_output_grid(
    tmp_path: Path,
    fixture_name: str,
    pattern: str,
    expected_resolution: int,
):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the real audio pipeline integration test")

    fixture_path = FIXTURE_DIR / fixture_name
    assert fixture_path.is_file(), f"Missing resolution fixture: {fixture_path}"
    ogg_path = convert_to_ogg(fixture_path, tmp_path / f"{fixture_path.stem}.ogg")
    analyzed = analyze_audio(ogg_path)
    raw = apply_analysis_overrides(analyzed, bpm=120.0, offset=0.5)

    beat_interval = 0.5
    onset_times: list[float] = []
    for bar_index in range(4):
        bar_start = 0.5 + bar_index * beat_interval * 4
        subdivisions = (
            4
            if pattern == "straight" or (pattern == "mixed" and bar_index % 2 == 0)
            else 3
        )
        for beat_index in range(4):
            beat_start = bar_start + beat_index * beat_interval
            onset_times.extend(
                beat_start + subdivision_index * beat_interval / subdivisions
                for subdivision_index in range(subdivisions)
            )
    raw = raw.model_copy(update={"onset_times": onset_times, "onset_strengths": []})

    features = build_bar_features(raw, max_bars=4)
    plan = build_resolution_plan(raw, features)
    chart_bars = generate_fallback_chart_bars(
        features,
        density="high",
        course="Oni",
        level=10,
        resolution_plan=plan,
    )

    assert raw.rms_envelope
    assert plan.base_resolution == expected_resolution
    assert plan.bar_resolutions == [expected_resolution] * len(features)
    assert all(len(chart_bar.notes) == expected_resolution for chart_bar in chart_bars)
    chart = TjaChart(
        metadata=ChartMetadata(
            title=f"Resolution {pattern}",
            wave=ogg_path.name,
            bpm=raw.bpm,
            offset=raw.offset,
        ),
        bars=chart_bars,
    )
    assert "#END" in render_tja(chart)


def test_structure_audio_fixture_detects_build_up_peak_and_drop(tmp_path: Path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the real audio pipeline integration test")

    fixture_path = FIXTURE_DIR / "structure_build_up_120.wav"
    assert fixture_path.is_file(), f"Missing structure fixture: {fixture_path}"
    ogg_path = convert_to_ogg(fixture_path, tmp_path / "structure_build_up_120.ogg")
    analyzed = analyze_audio(ogg_path)
    raw = apply_analysis_overrides(analyzed, bpm=120.0, offset=0.5)
    beat_interval = 0.5
    subdivisions_by_bar = [1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 1, 1]
    onset_times: list[float] = []
    for bar_index, subdivisions in enumerate(subdivisions_by_bar):
        bar_start = 0.5 + bar_index * beat_interval * 4
        for beat_index in range(4):
            beat_start = bar_start + beat_index * beat_interval
            onset_times.extend(
                beat_start + subdivision_index * beat_interval / subdivisions
                for subdivision_index in range(subdivisions)
            )
    raw = raw.model_copy(update={"onset_times": onset_times, "onset_strengths": []})

    structure = analyze_song_structure(build_bar_features(raw, max_bars=12))
    roles = [item.transition_role for item in structure.bar_structures]

    assert len(structure.bars) == 12
    assert len(structure.phrases) >= 2
    assert "build_up" in roles
    assert "peak" in roles
    assert "drop" in roles
    assert structure.bars[3].phrase_position != "phrase_end"
    assert structure.bars[7].phrase_position != "phrase_end"


def test_transient_noise_intro_is_forced_to_edge_silence(tmp_path: Path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the real audio pipeline integration test")

    fixture_path = FIXTURE_DIR / "transient_noise_intro_120.wav"
    assert fixture_path.is_file(), f"Missing transient noise fixture: {fixture_path}"
    ogg_path = convert_to_ogg(fixture_path, tmp_path / "transient_noise_intro_120.ogg")
    analyzed = analyze_audio(ogg_path)
    raw = apply_analysis_overrides(analyzed, bpm=120.0, offset=0.0)
    raw = raw.model_copy(
        update={
            "onset_times": [0.5, 1.5]
            + [onset_time for onset_time in raw.onset_times if onset_time >= 2.0]
        }
    )

    features = build_bar_features(raw)
    hints = build_density_hints(features)
    chart_bars = generate_fallback_chart_bars(features, density="max", course="Oni", level=10)
    report = build_quality_report(chart_bars, features)

    assert raw.rms_envelope
    assert features[0].rms_dbfs is not None and features[0].rms_dbfs <= -50.0
    assert features[0].peak_rms_dbfs is not None and features[0].peak_rms_dbfs <= -40.0
    assert features[0].relative_rms_db is not None and features[0].relative_rms_db <= -40.0
    assert features[0].sustained_activity_ratio is not None
    assert features[0].sustained_activity_ratio <= 0.15
    assert features[0].onset_grids
    assert hints[0].kind == "silent"
    assert chart_bars[0].notes == "0" * 16
    assert hints[1].kind != "silent"
    assert playable_hit_count(chart_bars[1].notes) > 0
    assert report.silent_bar_note_count == 0

    chart = TjaChart(
        metadata=ChartMetadata(
            title="Transient Noise Intro",
            wave=ogg_path.name,
            bpm=raw.bpm,
            offset=raw.offset,
        ),
        bars=chart_bars,
    )
    assert "#END" in render_tja(chart)
