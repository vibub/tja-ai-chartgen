from tja_ai_chartgen.ai.examples import get_reference_examples_prompt


def test_reference_examples_prompt_is_static_and_path_free():
    prompt = get_reference_examples_prompt()

    assert "Reference chart examples" in prompt
    assert "reference_notes" in prompt
    assert "audio_features" not in prompt
    assert "D:/Downloads" not in prompt
    assert "D:\\Downloads" not in prompt
    assert "100000200200200000100000000000100000200000200000" in prompt
