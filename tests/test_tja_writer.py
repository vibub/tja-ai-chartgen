from tja_ai_chartgen.tja.model import ChartBar, ChartMetadata, TjaChart
from tja_ai_chartgen.tja.writer import render_tja


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
    assert "#START" in text
    assert "#END" in text
    assert "1000100010001000," in text


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
