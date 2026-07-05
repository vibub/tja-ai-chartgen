import pytest

from tja_ai_chartgen.tja.model import ChartBar, ChartMetadata, TjaChart
from tja_ai_chartgen.tja.writer import TJA_FILE_ENCODING, TjaEncodingError, read_tja_text, render_tja, write_tja_text


def test_render_tja_outputs_required_sections():
    chart = TjaChart(
        metadata=ChartMetadata(
            title="Song Title",
            artist="Artist Name",
            wave="song.ogg",
            bpm=174.0,
            offset=-0.032,
            course="Oni",
            level=10,
        ),
        bars=[ChartBar(index=0, notes="1000100010001000")],
    )

    text = render_tja(chart)

    assert "TITLE:Song Title" in text
    assert "SUBTITLE:-- Artist Name" in text
    assert "OFFSET:0.032" in text
    assert "#START" in text
    assert "#END" in text
    assert "1000100010001000," in text


def test_render_tja_writes_tja_offset_with_opposite_sign():
    chart = TjaChart(
        metadata=ChartMetadata(
            title="Song Title",
            wave="song.ogg",
            bpm=120.0,
            offset=0.725,
        ),
        bars=[ChartBar(index=0, notes="1000100010001000")],
    )

    text = render_tja(chart)

    assert "OFFSET:-0.725" in text


def test_render_tja_outputs_measure_for_three_four_bars():
    chart = TjaChart(
        metadata=ChartMetadata(
            title="Song Title",
            wave="song.ogg",
            bpm=120.0,
            offset=0.0,
        ),
        bars=[ChartBar(index=0, notes="100010001000", time_signature="3/4")],
    )

    text = render_tja(chart)

    assert "#MEASURE 3/4\n100010001000," in text
    assert "100010001000,\n#MEASURE 1/1\n#END" in text


def test_render_tja_outputs_balloon_header():
    chart = TjaChart(
        metadata=ChartMetadata(
            title="Song Title",
            wave="song.ogg",
            bpm=120.0,
            offset=0.0,
        ),
        bars=[ChartBar(index=0, notes="7000000080000000", balloon_counts=[8])],
    )

    text = render_tja(chart)

    assert "BALLOON:8" in text
    assert "7000000080000000," in text


def test_write_tja_text_saves_shift_jis_compatible_file(tmp_path):
    path = tmp_path / "song.tja"
    text = "TITLE:迷っちゃうわ\nWAVE:mayocchauwa.ogg\n#START\n1000,\n#END\n"

    write_tja_text(path, text)

    assert path.read_bytes() == text.encode(TJA_FILE_ENCODING)
    assert read_tja_text(path) == text


def test_write_tja_text_normalizes_decomposed_dakuten(tmp_path):
    path = tmp_path / "song.tja"
    text = "TITLE:が\nWAVE:ga.ogg\n#START\n1000,\n#END\n"

    write_tja_text(path, text)

    assert read_tja_text(path).startswith("TITLE:が\n")
    assert path.read_bytes().startswith("TITLE:が".encode(TJA_FILE_ENCODING))


def test_write_tja_text_reports_shift_jis_incompatible_character(tmp_path):
    path = tmp_path / "song.tja"
    text = "TITLE:乌鸦\nWAVE:karasu.ogg\n#START\n1000,\n#END\n"

    with pytest.raises(TjaEncodingError) as error:
        write_tja_text(path, text)

    message = str(error.value)
    assert "cp932" in message
    assert "line 1, column 7" in message
    assert "'乌'" in message
    assert "U+4E4C" in message
    assert not path.exists()
