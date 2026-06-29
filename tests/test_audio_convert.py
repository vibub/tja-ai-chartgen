from tja_ai_chartgen.audio.convert import convert_to_ogg


class FakeResult:
    returncode = 0
    stdout = ""
    stderr = ""


def test_convert_to_ogg_calls_ffmpeg(tmp_path, monkeypatch):
    input_path = tmp_path / "song.mp3"
    input_path.write_bytes(b"fake audio")
    output_path = tmp_path / "out" / "song.ogg"

    calls = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        output_path.write_bytes(b"fake ogg")
        return FakeResult()

    monkeypatch.setattr("subprocess.run", fake_run)

    result = convert_to_ogg(input_path, output_path)

    assert result == output_path
    assert output_path.exists()
    assert calls
    assert calls[0][0] == "ffmpeg"


def test_convert_to_ogg_raises_for_missing_input(tmp_path):
    output_path = tmp_path / "out" / "song.ogg"

    try:
        convert_to_ogg(tmp_path / "missing.mp3", output_path)
    except FileNotFoundError as error:
        assert "Input audio file not found" in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError")
