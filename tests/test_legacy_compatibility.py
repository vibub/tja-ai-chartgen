import json
from pathlib import Path

import pytest

from tja_ai_chartgen.ai.prompts import build_chart_generation_payload
from tja_ai_chartgen.ai.sidecars import load_ai_sidecar
from tja_ai_chartgen.audio.instruments import InstrumentAnalysisRaw
from tja_ai_chartgen.generation import load_generation_config
from tja_ai_chartgen.tja.model import SongAnalysis


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "compatibility"


def test_legacy_instrument_v1_partial_snapshot_preserves_original_semantics():
    raw = InstrumentAnalysisRaw.model_validate_json(
        (FIXTURE_DIR / "instrument_v1_partial.json").read_text(encoding="utf-8")
    )

    assert raw.feature_version == "instrument-v1"
    assert raw.status == "partial"
    assert raw.reason == "classifier-load-error:OSError"
    assert raw.classifier_model == "MIT/ast-finetuned-audioset-10-10-0.4593"
    assert raw.has_stem_evidence is True
    assert raw.has_classifier_evidence is False
    assert raw.uses_legacy_instrument_semantics is True
    assert raw.model_dump(mode="json")["status"] == "partial"


def test_legacy_analysis_snapshot_reads_old_aliases_and_concrete_instrument_fields():
    analysis = SongAnalysis.model_validate_json(
        (FIXTURE_DIR / "analysis_v5_instrument_v1.json").read_text(encoding="utf-8")
    )

    assert analysis.analysis_schema_version == 5
    assert analysis.instrument_feature_version == "instrument-v1"
    assert analysis.instrument_analysis_status == "partial"
    assert analysis.instrument_analysis_reason == "classifier-load-error:OSError"
    assert analysis.bars[0].onset_grids == [0, 4, 8, 12]
    assert analysis.bars[0].accent_grids == [0, 8]
    assert analysis.bars[0].instrument.guitar == 0.75
    assert analysis.bars[0].instrument.dominant_instrument == "guitar"
    assert analysis.bar_structures == []
    assert analysis.phrase_plan == []
    assert analysis.resolution_plan is None

    payload = build_chart_generation_payload(
        analysis,
        "Oni",
        10,
        "technical",
        reference_examples_prompt="[]",
    )

    assert payload["schema"] == "tja-ai-chartgen-compact-v7"
    assert payload["instrument_feature_version"] == "instrument-v1"
    assert "dominant_instrument" not in payload["legend"]["instrument_bar_columns"]
    assert "active_instruments" not in payload["legend"]["instrument_bar_columns"]


def test_legacy_generation_config_snapshot_defaults_to_full_profile_and_drops_secrets():
    config = load_generation_config(FIXTURE_DIR / "generation_config_v0.json")

    assert config.schema_version == 1
    assert config.use_beatnet is True
    assert config.use_instrument_analysis is True
    assert config.instrument_profile == "full"
    assert config.instrument_device == "cpu"
    assert config.instrument_model_dir == Path("models/instrument-v1")
    assert config.model == "openai/legacy-model"
    payload = config.model_dump(mode="json")
    assert "ai_base_url" not in payload
    assert "ai_api_key" not in payload


@pytest.mark.parametrize(
    ("filename", "expected_schema"),
    [
        ("ai_input_compact_v4.json", "tja-ai-chartgen-compact-v4"),
        ("ai_output_legacy.json", None),
        ("ai_attempts_legacy.json", None),
    ],
)
def test_legacy_ai_sidecar_snapshots_load_without_schema_rewrite(filename, expected_schema):
    payload = load_ai_sidecar(FIXTURE_DIR / filename)

    assert payload.get("schema") == expected_schema
    if filename == "ai_input_compact_v4.json":
        assert payload["instrument_feature_version"] == "instrument-v1"
        assert payload["bar_instruments"][0][6] == "guitar"
        assert payload["bar_instruments"][0][8] == [["guitar", 750], ["synth", 400]]
    elif filename == "ai_output_legacy.json":
        assert payload["final"]["bars"][0]["notes"] == "1000100010001000"
    else:
        assert payload["attempts"][0]["status"] == "validation_error"
        assert payload["fallback_reason"] == "content_repair_exhausted"
    assert "rhythm_repair_gate" not in payload
    assert "salience_validation" not in payload


def test_ai_sidecar_loader_rejects_non_object_json(tmp_path):
    sidecar_path = tmp_path / "ai_output.json"
    sidecar_path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="AI sidecar must be a JSON object"):
        load_ai_sidecar(sidecar_path)
