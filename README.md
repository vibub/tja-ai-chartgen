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

Each `generate` run writes `generation_config.json` next to the TJA output. It records the input path, metadata, difficulty, style, density, `--max-bars`, BPM/OFFSET overrides, AI flag, and model name so a useful draft can be reproduced later.

## Limitations

- Best for songs with stable BPM.
- Assumes 4/4 time signature.
- Generated charts are drafts and require human review.
- OFFSET may need manual adjustment in OpenTaiko or another simulator.
- `--max-bars` is intended for quick draft checks and truncates the generated chart to the first N bars.
- `--bpm` and `--offset` override automatic analysis results for manual calibration.
- `--density auto|low|medium|high|max` controls rule-based draft density and is passed into the AI prompt when `--use-ai` is enabled.
- MVP does not support BPM changes, branches, drumrolls, balloons, or scroll gimmicks.
