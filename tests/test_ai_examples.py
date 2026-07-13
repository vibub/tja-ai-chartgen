import json
import re

from tja_ai_chartgen.ai.examples import get_reference_examples_prompt
from tja_ai_chartgen.ai.reference_windows import get_reference_windows_payload


def test_reference_examples_prompt_is_static_diverse_and_path_free():
    prompt = get_reference_examples_prompt()
    payload = json.loads(re.search(r"\[\s*\{.*\}\s*\]", prompt, re.DOTALL).group(0))

    assert "Reference chart examples" in prompt
    assert "evenly sampled across each full chart" in prompt
    assert "reference_notes" in prompt
    assert "audio_features" not in prompt
    assert "D:/Downloads" not in prompt
    assert "D:\\Downloads" not in prompt
    assert len(payload) == 3
    assert sum(len(example["bars"]) for example in payload) == 72
    assert {example["sample_strategy"] for example in payload} == {
        "24 evenly sampled bars across the full chart"
    }
    assert any(bar["section"] != "intro" for example in payload for bar in example["bars"])
    assert len({bar["phrase_position"] for example in payload for bar in example["bars"]}) > 2


def test_reference_windows_are_continuous_course_aware_and_path_free():
    payload = get_reference_windows_payload("Oni", 9)

    assert payload is not None
    assert len(payload["windows"]) == 3
    assert {window["role"] for window in payload["windows"]} == {
        "intro",
        "peak",
        "cadence",
    }
    assert all(window["course"] == "Oni" for window in payload["windows"])
    assert all(abs(window["level"] - 9) <= 2 for window in payload["windows"])
    assert all(
        [bar[0] for bar in window["bars"]] == list(range(len(window["bars"])))
        for window in payload["windows"]
    )
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "D:/Downloads" not in serialized
    assert "D:\\Downloads" not in serialized
    assert "TITLE" not in serialized
    assert "WAVE" not in serialized
