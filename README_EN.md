# tja-ai-chartgen

AI-assisted TJA chart draft generator for Taiko simulators.

## Requirements

- Python 3.11+
- ffmpeg

## Install

```bash
pip install -e ".[dev]"
```

## Usage

Check the installed CLI version:

```bash
tja-ai-chartgen version
```

Rule-based draft:

```bash
tja-ai-chartgen generate song.mp3 --title "Song Title"
```

Generate only the first N bars for quick checks:

```bash
tja-ai-chartgen generate song.mp3 --title "Song Title" --max-bars 16
```

Control rule-based draft density:

```bash
tja-ai-chartgen generate song.mp3 --title "Song Title" --density high
```

Allowed density values are `auto`, `low`, `medium`, `high`, and `max`. `auto` follows analyzed bar energy.

Override analyzed BPM and OFFSET when manual calibration is needed:

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --max-bars 16 \
  --bpm 220.588 \
  --offset 0.725
```

AI-assisted draft:

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --course Oni \
  --level 10 \
  --style technical \
  --use-ai
```

AI calls are routed through LiteLLM. By default the CLI reads `MODEL`; you can also pass `--model` explicitly. For OpenAI-compatible endpoints, model names usually use LiteLLM's `openai/` prefix and can be combined with a custom base URL and API key:

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --use-ai \
  --model openai/custom-model \
  --ai-base-url https://llm.example.com/v1 \
  --ai-api-key sk-...
```

You can also configure these values in `.env` to avoid writing API keys into shell history:

```env
MODEL=openai/custom-model
OPENAI_BASE_URL=https://llm.example.com/v1
OPENAI_API_KEY=sk-...
```

AI output is strictly validated before use. If JSON shape, bar count, note length, or allowed characters fail MVP constraints, the CLI sends a repair prompt and retries. The default repair retry count is 2 and can be changed with `--ai-repair-retries`. If repair still fails, generation falls back to the rule-based generator.

Re-run a saved generation config:

```bash
tja-ai-chartgen generate-from-config output/generation_config.json
```

## Output

```txt
output/
├─ song.ogg
├─ song.tja
├─ analysis.json
├─ generation_config.json
├─ ai_input.json
├─ ai_output.json
└─ report.txt
```

## Reproducibility

Each `generate` run writes `generation_config.json` next to the TJA output. It records the input path, metadata, difficulty, style, density, `--max-bars`, BPM/OFFSET overrides, AI flag, final resolved model name, and AI repair retry count so a useful draft can be reproduced later. Use `generate-from-config` to run the same generation parameters again. API keys and base URLs are not written to `generation_config.json`; provide connection settings again through `.env`, environment variables, or command options when rerunning AI generation.

## Limitations

- Best for songs with stable BPM.
- Assumes 4/4 time signature.
- Generated charts are drafts and require human review.
- OFFSET may need manual adjustment in OpenTaiko or another simulator.
- `--max-bars` is intended for quick draft checks and truncates the generated chart to the first N bars.
- `--bpm` and `--offset` override automatic analysis results for manual calibration.
- `--density auto|low|medium|high|max` controls rule-based draft density and is passed into the AI prompt when `--use-ai` is enabled.
- `--use-ai` can still fall back to the rule-based generator when the model is unavailable, output repair is exhausted, or credentials are misconfigured.
- MVP does not support BPM changes, branches, drumrolls, balloons, or scroll gimmicks.

## Roadmap

These items are planned for later versions and are not part of the MVP scope.

### v0.2

- Support manual BPM override. ✅ Implemented in MVP iteration via `--bpm`.
- Support manual OFFSET override. ✅ Implemented in MVP iteration via `--offset`.
- Support `--max-bars` to generate only the first N bars for easier testing. ✅ Implemented in MVP iteration.
- Add automatic AI output repair and retry. ✅ Implemented; defaults to 2 repair retries and can be adjusted with `--ai-repair-retries`.
- Add `--density low|medium|high|max`. ✅ Implemented in MVP iteration via `--density auto|low|medium|high|max`.

### v0.3

- Try integrating BeatNet for:
  - downbeat detection
  - meter detection
  - more accurate bar starts
- Support `3/4` and `6/8` time signatures.
- Support simple drumrolls and balloons.

### v0.4

- Support multiple difficulties:
  - Easy
  - Normal
  - Hard
  - Oni
- Support style templates:
  - technical
  - stamina
  - hybrid
  - performance

### v0.5

- Build a Web UI.
- Upload audio files.
- Preview analysis results online.
- Manually adjust BPM / OFFSET.
- Regenerate selected bars.
