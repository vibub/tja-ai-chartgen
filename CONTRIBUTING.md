# Contributing

Thanks for considering a contribution to `tja-ai-chartgen`.

## Development Setup

Requirements:

- Python 3.11+
- ffmpeg

Install the project with development dependencies:

```bash
pip install -e ".[dev]"
```

## Local Checks

Run these before opening a pull request:

```bash
ruff check .
pytest
```

For focused TJA writer checks:

```bash
pytest tests/test_tja_writer.py -v
```

For the real ffmpeg, librosa, bar-feature, and TJA export pipeline:

```bash
pytest tests/test_audio_pipeline_integration.py -v
```

## Golden Audio Fixtures

The WAV files under `tests/fixtures/audio/` are deterministic click tracks generated entirely by project code. They contain no third-party recordings or copyrighted music. Rebuild them with:

```bash
python tests/fixtures/audio/rebuild_click_fixtures.py
```

When adding or changing an audio fixture, keep it small, use a clearly licensed or programmatically generated source, and update the rebuild script and expected integration-test parameters together. Do not replace these fixtures with copyrighted songs or samples.

## Contribution Scope

This project is an MVP for AI-assisted `.tja` chart draft generation. Contributions are easiest to review when they are focused on one behavior or one documentation improvement at a time.

Please avoid including generated output, local `.env` files, cache directories, or copyrighted audio files in pull requests.

## AI and API Keys

Do not commit API keys. Use `.env`, environment variables, or command-line options for local AI generation tests. Generated configs intentionally do not persist API keys.
