import json
import shutil
import wave
from array import array

import pytest
from typer.testing import CliRunner

from tja_ai_chartgen.audio.timing import (
    build_timing_diagnostic,
    fit_timing_anchors,
    parse_timing_anchors,
    render_timing_markdown,
)
from tja_ai_chartgen.audio.timing_preview import write_timing_previews
from tja_ai_chartgen.cli import app
from tja_ai_chartgen.generation import GenerationConfig, load_generation_config
from tja_ai_chartgen.tja.model import BarFeature, SongAnalysis, TempoMeterCandidate
from tja_ai_chartgen.utils.paths import write_json


def _analysis(times=None, *, tracker_bpm=120, source="librosa", duration=60):
    return SongAnalysis(
        title="Timing", audio_file="song.wav", ogg_file="song.ogg", bpm=120, offset=0,
        bars=[BarFeature(index=0, start_time=0, end_time=duration, energy=.5)],
        tempo_candidates=[] if times is None else [TempoMeterCandidate(
            source=source, bpm=tracker_bpm, offset=0, time_signature="4/4", beat_times=times,
        )],
    )


@pytest.mark.parametrize("delay,classification", [
    (0, "aligned-to-tracker"), (.08, "possible-offset-shift"),
])
def test_constant_shift_is_distinguished_from_drift(delay, classification):
    report = build_timing_diagnostic(_analysis([i * .5 + delay for i in range(120)]))
    tracker = report["trackers"][0]
    assert tracker["classification"] == classification
    assert all(w["median_error_seconds"] == pytest.approx(delay) for w in tracker["windows"])


def test_cumulative_drift_does_not_wrap_back_after_half_a_beat():
    times = [i * 60 / 118 for i in range(118)]
    tracker = build_timing_diagnostic(_analysis(times, tracker_bpm=118))["trackers"][0]
    assert tracker["classification"] == "possible-bpm-drift"
    assert tracker["points"][-1]["error_seconds"] > .9


def test_missed_and_duplicate_beats_do_not_shift_all_later_events():
    times = [i * .5 for i in range(120) if i != 40] + [10.02]
    tracker = build_timing_diagnostic(_analysis(times))["trackers"][0]
    assert tracker["classification"] == "aligned-to-tracker"
    assert tracker["ignored_duplicate_count"] == 1
    assert tracker["points"][-1]["error_seconds"] == pytest.approx(0)


@pytest.mark.parametrize("source", ["librosa+onset-grid", "beatnet+onset-grid"])
def test_generated_grid_is_never_accepted_as_independent_evidence(source):
    report = build_timing_diagnostic(_analysis([i * .5 for i in range(120)], source=source))
    assert report["status"] == "no-independent-tracker"
    assert report["trackers"] == []


def test_short_and_legacy_analysis_report_insufficient_evidence():
    assert build_timing_diagnostic(_analysis())["trackers"] == []
    report = build_timing_diagnostic(_analysis([0, .5, 1], duration=60))
    assert report["trackers"][0]["classification"] == "insufficient-evidence"


def test_half_tempo_tracker_is_explicitly_ambiguous():
    tracker = build_timing_diagnostic(_analysis(list(range(60)), tracker_bpm=60))["trackers"][0]
    assert tracker["classification"] == "tempo-mismatch-or-alias"


def test_local_tracker_jump_is_not_presented_as_a_simple_bpm_fix():
    times = [i * .5 + (.1 if 40 <= i < 80 else 0) for i in range(120)]
    tracker = build_timing_diagnostic(_analysis(times))["trackers"][0]
    assert tracker["classification"] == "irregular-tracker"
    assert "suggested_bpm" not in tracker


def test_manual_anchors_fix_bpm_and_offset_including_negative_origin():
    fit = fit_timing_anchors([(1, .25), (65, 32.25), (129, 64.25)])
    assert fit["bpm"] == pytest.approx(120)
    assert fit["offset"] == pytest.approx(-.25)
    assert fit["tja_offset"] == pytest.approx(.25)
    assert fit["consistent"] and fit["independently_checked"]
    analysis = _analysis().model_copy(update={"bpm": 119, "offset": .1, "time_signature": "6/8"})
    report = build_timing_diagnostic(analysis, anchors=[(1, .25), (65, 32.25), (129, 64.25)])
    assert report["anchor_comparison"]["before"]["p95_absolute_error_seconds"] > .5
    assert report["anchor_comparison"]["after"]["p95_absolute_error_seconds"] < 1e-9


def test_two_anchors_do_not_claim_independent_validation():
    report = build_timing_diagnostic(_analysis(), anchors=[(0, .25), (64, 32.25)])
    assert not report["calibration"]["independently_checked"]
    assert "只有两个锚点" in render_timing_markdown(report)


def test_middle_anchor_exposes_nonconstant_tempo_or_bad_counting():
    fit = fit_timing_anchors([(0, 0), (64, 33), (128, 64)])
    assert not fit["consistent"]
    assert fit["max_fit_error_seconds"] > .6


@pytest.mark.parametrize("anchors", [
    [(0, 0)], [(0, 0), (0, 1)], [(0, 1), (1, 0)],
    [(0, float("nan")), (1, 1)], [(0, -1), (1, 1)],
])
def test_invalid_manual_anchors_are_rejected(anchors):
    with pytest.raises(ValueError):
        fit_timing_anchors(anchors)


def test_invalid_anchor_syntax_is_actionable():
    with pytest.raises(ValueError, match="BEAT:SECONDS"):
        parse_timing_anchors(["0,0.25"])


def test_cli_exports_separate_calibrated_config_and_keeps_original(tmp_path):
    analysis_path = tmp_path / "analysis.json"
    write_json(analysis_path, _analysis())
    config_path = tmp_path / "generation_config.json"
    write_json(config_path, GenerationConfig(input_audio=tmp_path / "song.wav", title="Timing"))
    before = config_path.read_bytes(), analysis_path.read_bytes()
    result = CliRunner().invoke(app, [
        "diagnose-timing", str(analysis_path), "--config", str(config_path),
        "--anchor", "0:0.25", "--anchor", "64:32.25", "--anchor", "112:56.25",
    ])
    assert result.exit_code == 0, result.output
    exported = load_generation_config(tmp_path / "timing/generation_config.calibrated.json")
    assert exported.bpm_override == pytest.approx(120)
    assert exported.offset_override == pytest.approx(.25)
    assert exported.output_dir == tmp_path / "timing/calibrated"
    assert (config_path.read_bytes(), analysis_path.read_bytes()) == before
    saved = json.loads((tmp_path / "timing/timing_diagnostic.json").read_text())
    assert saved["diagnostic_version"] == "timing-diagnostic-v1"


def test_inconsistent_anchors_cannot_export_a_generation_config(tmp_path):
    analysis_path = tmp_path / "analysis.json"
    config_path = tmp_path / "generation_config.json"
    write_json(analysis_path, _analysis())
    write_json(config_path, GenerationConfig(input_audio=tmp_path / "song.wav", title="Timing"))
    result = CliRunner().invoke(app, [
        "diagnose-timing", str(analysis_path), "--config", str(config_path),
        "--anchor", "0:0", "--anchor", "64:33", "--anchor", "128:64",
    ])
    assert result.exit_code != 0
    assert "consistent manual anchors" in result.output
    assert not (tmp_path / "timing").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires ffmpeg")
def test_audio_ab_previews_place_clicks_on_each_clock(tmp_path):
    audio = tmp_path / "silence.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(22050)
        stream.writeframes(b"\0\0" * 44100)
    analysis = _analysis(duration=2)
    report = build_timing_diagnostic(analysis, anchors=[(0, .1), (3, 1.6)])
    previews = write_timing_previews(audio, tmp_path / "clips", analysis, report)
    assert len(previews) == 2
    samples = []
    for preview in previews:
        with wave.open(str(tmp_path / "clips" / preview["file"]), "rb") as stream:
            assert stream.getframerate() == 22050
            samples.append(array("h", stream.readframes(stream.getnframes())))
    assert max(abs(x) for x in samples[0][:441]) > 1000
    assert not any(samples[1][:441])
    assert max(abs(x) for x in samples[1][2205:2646]) > 1000
    assert not any(samples[0][2205:2646])
