from pathlib import Path
import shutil

import pytest

from tja_ai_chartgen.audio.analyze import analyze_audio
from tja_ai_chartgen.audio.convert import convert_to_ogg
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.tja.model import ChartBar, ChartMetadata, TjaChart
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
    assert len(raw.onset_times) >= 12
    assert raw.beat_times
    assert raw.analyzer in {"librosa", "librosa+onset-grid"}

    features = build_bar_features(raw)

    assert features
    assert all(bar.time_signature == "4/4" for bar in features)
    assert all(bar.grids_per_bar == 16 for bar in features)
    assert any(bar.onset_16 for bar in features)

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
