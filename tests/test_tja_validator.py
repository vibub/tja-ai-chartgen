from tja_ai_chartgen.tja.validator import validate_tja_text


def test_validate_tja_text_accepts_basic_chart():
    text = """TITLE:Song Title
BPM:174.0
WAVE:song.ogg
OFFSET:-0.032
COURSE:Oni
LEVEL:10

#START
1000100010001000,
#END
"""

    issues = validate_tja_text(text)

    assert not [issue for issue in issues if issue.level == "error"]


def test_validate_tja_text_reports_missing_header():
    text = """TITLE:Song Title
#START
1000,
#END
"""

    issues = validate_tja_text(text)

    assert any(issue.level == "error" and "Missing required header: BPM:" in issue.message for issue in issues)


def test_validate_tja_text_warns_for_bar_without_comma():
    text = """TITLE:Song Title
BPM:174.0
WAVE:song.ogg
OFFSET:-0.032
COURSE:Oni
LEVEL:10

#START
1000100010001000
#END
"""

    issues = validate_tja_text(text)

    assert any(issue.level == "warning" and issue.line == 9 for issue in issues)
