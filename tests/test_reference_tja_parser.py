from pathlib import Path

import pytest

from tja_ai_chartgen.reference.tja_parser import (
    ReferenceTjaParseError,
    parse_tja_file,
    parse_tja_text,
)


MULTI_COURSE_TJA = """\
TITLE:Reference Song
BPM:120
WAVE:reference.ogg
OFFSET:-1.0

COURSE:Oni
LEVEL:8
BALLOON:5
#START
1000,
#GOGOSTART
#BPMCHANGE 240
#MEASURE 3/4
700800100000,
#GOGOEND
#BARLINEOFF
0101,0011,
#BARLINEON
#END

COURSE:Easy
LEVEL:3
#START
1000,0001,
#END
"""


def test_parse_tja_text_builds_independent_course_timelines():
    parsed = parse_tja_text(MULTI_COURSE_TJA, source_id="sample")

    assert parsed.source_id == "sample"
    assert parsed.title == "Reference Song"
    assert parsed.wave == "reference.ogg"
    assert parsed.bpm == 120.0
    assert parsed.internal_offset == 1.0
    assert [course.course for course in parsed.courses] == ["Oni", "Easy"]
    assert [course.level for course in parsed.courses] == [8, 3]

    oni = parsed.courses[0]
    assert [bar.notes for bar in oni.bars] == ["1000", "700800100000", "0101", "0011"]
    assert [bar.resolution for bar in oni.bars] == [4, 12, 4, 4]
    assert oni.bars[0].start_time == pytest.approx(1.0)
    assert oni.bars[0].end_time == pytest.approx(3.0)
    assert oni.bars[1].start_time == pytest.approx(3.0)
    assert oni.bars[1].end_time == pytest.approx(3.75)
    assert oni.bars[1].bpm == 240.0
    assert oni.bars[1].measure_ratio == "3/4"
    assert oni.bars[1].gogo is True
    assert oni.bars[1].balloon_counts == [5]
    assert oni.bars[2].barline_visible is False
    assert oni.bars[3].barline_visible is False

    easy = parsed.courses[1]
    assert [bar.start_time for bar in easy.bars] == pytest.approx([1.0, 3.0])
    assert [bar.end_time for bar in easy.bars] == pytest.approx([3.0, 5.0])
    assert all(bar.bpm == 120.0 for bar in easy.bars)
    assert all(bar.measure_ratio == "1/1" for bar in easy.bars)


def test_parse_tja_text_supports_big_roll_and_kusudama_reference_notes():
    parsed = parse_tja_text(
        "BPM:120\nCOURSE:Oni\nLEVEL:8\nBALLOON:9\n#START\n6008,9008,\n#END\n",
        source_id="special-notes",
    )

    assert [bar.notes for bar in parsed.courses[0].bars] == ["6008", "9008"]
    assert parsed.courses[0].bars[0].balloon_counts == []
    assert parsed.courses[0].bars[1].balloon_counts == [9]


def test_parse_tja_text_supports_bpm_change_inside_measure():
    notes_before_change = "1" + ("0" * 35)
    notes_after_change = "2" + ("0" * 23)
    parsed = parse_tja_text(
        "\n".join(
            [
                "BPM:100",
                "COURSE:Edit",
                "LEVEL:8",
                "#START",
                "#MEASURE 5/4",
                notes_before_change,
                "#BPMCHANGE 200",
                notes_after_change + ",",
                "#END",
            ]
        ),
        source_id="mid-bar-bpm",
    )

    bar = parsed.courses[0].bars[0]
    assert bar.resolution == 60
    assert bar.bpm == 100.0
    assert [(change.position, change.bpm) for change in bar.tempo_changes] == [(36, 200.0)]
    assert bar.start_time == pytest.approx(0.0)
    assert bar.end_time == pytest.approx(2.4)


def test_parse_tja_file_decodes_cp932_and_uses_content_hash_source_id(tmp_path: Path):
    path = tmp_path / "reference.tja"
    path.write_bytes(MULTI_COURSE_TJA.replace("Reference Song", "千本桜").encode("cp932"))

    first = parse_tja_file(path)
    second = parse_tja_file(path)

    assert first.title == "千本桜"
    assert first.source_id == second.source_id
    assert first.source_id != path.stem
    assert len(first.source_id) == 16


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ("BPM:120\nCOURSE:Oni\nLEVEL:8\n#START\n1000,\n", "unclosed-chart"),
        ("BPM:0\nCOURSE:Oni\nLEVEL:8\n#START\n1000,\n#END\n", "invalid-bpm"),
        (
            "BPM:120\nCOURSE:Oni\nLEVEL:8\n#START\n#MEASURE 0/4\n1000,\n#END\n",
            "invalid-measure",
        ),
        (
            "BPM:120\nCOURSE:Oni\nLEVEL:8\n#START\n#GOGOEND\n1000,\n#END\n",
            "orphan-gogo-end",
        ),
        (
            "BPM:120\nCOURSE:Oni\nLEVEL:8\n#START\n#BRANCHSTART p,50,100\n1000,\n#END\n",
            "unsupported-command",
        ),
        (
            "BPM:120\nCOURSE:Oni\nLEVEL:8\n#START\nX000,\n#END\n",
            "unsupported-note-character",
        ),
        (
            "BPM:120\nCOURSE:Oni\nLEVEL:8\nBALLOON:5\n#START\n7008,7008,\n#END\n",
            "balloon-count-mismatch",
        ),
        (
            "BPM:120\nCOURSE:Oni\nLEVEL:8\n#START\n10\n#MEASURE 3/4\n00,\n#END\n",
            "command-inside-bar",
        ),
    ],
)
def test_parse_tja_text_rejects_unreliable_timing_or_chart_data(body: str, code: str):
    with pytest.raises(ReferenceTjaParseError) as error:
        parse_tja_text(body, source_id="invalid")

    assert error.value.code == code
    assert error.value.line is not None
