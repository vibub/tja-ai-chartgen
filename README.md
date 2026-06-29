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
├─ ai_input.json
├─ ai_output.json
└─ report.txt
```

## Limitations

- Best for songs with stable BPM.
- Assumes 4/4 time signature.
- Generated charts are drafts and require human review.
- OFFSET may need manual adjustment in OpenTaiko or another simulator.
- MVP does not support BPM changes, branches, drumrolls, balloons, or scroll gimmicks.
