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

## Contribution Scope

This project is an MVP for AI-assisted `.tja` chart draft generation. Contributions are easiest to review when they are focused on one behavior or one documentation improvement at a time.

Please avoid including generated output, local `.env` files, cache directories, or copyrighted audio files in pull requests.

## AI and API Keys

Do not commit API keys. Use `.env`, environment variables, or command-line options for local AI generation tests. Generated configs intentionally do not persist API keys.
