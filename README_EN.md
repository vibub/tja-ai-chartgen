# tja-ai-chartgen

AI-assisted TJA chart draft generator for Taiko simulators.

Chinese documentation: [README.md](README.md).

License: MIT. See [LICENSE](LICENSE).

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

Generate a rule-based chart draft:

```bash
tja-ai-chartgen generate song.mp3 --title "Song Title"
```

Generate only the first N bars for quick checks:

```bash
tja-ai-chartgen generate song.mp3 --title "Song Title" --max-bars 16
```

Control rule-based draft density and style template:

```bash
tja-ai-chartgen generate song.mp3 --title "Song Title" --density high --style hybrid
```

Allowed `--density` values are `auto`, `low`, `medium`, `high`, and `max`. `auto` follows analyzed bar energy. Allowed `--style` values are `technical`, `stamina`, `hybrid`, and `performance`; they affect rule-based templates and AI prompts.

Allow the rule generator and AI to use simple drumroll and balloon notes:

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --density high \
  --special-notes
```

When enabled, the rule generator may place a small number of `5...8` drumrolls and `7...8` balloons in high-energy bars, and the writer emits the matching `BALLOON:` header.

Override analyzed BPM and OFFSET when manual calibration is needed:

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --max-bars 16 \
  --bpm 220.588 \
  --offset 0.725
```

Override or enhance time signature analysis. Supported values are `4/4`, `3/4`, and `6/8`:

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --time-signature 3/4
```

`3/4` and `6/8` use 12-grid bars and are exported with `#MEASURE 3/4` in the `.tja`; `4/4` keeps the default 16-grid bars.

Try optional BeatNet analysis for downbeat, meter, and bar-start enhancement:

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --use-beatnet
```

BeatNet is not a default dependency. Install `BeatNet` separately before using it. If BeatNet is unavailable or analysis fails, the CLI keeps the default librosa analysis result.

Generate Easy, Normal, Hard, and Oni charts in one run:

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --all-courses
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

AI calls are routed through LiteLLM using an OpenAI-compatible protocol. The recommended setup is to put model and connection settings in `.env` so API keys do not appear in shell history:

```env
MODEL=openai/custom-model
OPENAI_BASE_URL=https://llm.example.com/v1
OPENAI_API_KEY=sk-...
```

Variable meanings:

- `MODEL`: model name to call. It can be the model exposed by your compatible gateway, or a LiteLLM provider-prefixed name such as `openai/custom-model`.
- `OPENAI_BASE_URL`: OpenAI-compatible endpoint URL, for example `https://llm.example.com/v1`.
- `OPENAI_API_KEY`: API key for the OpenAI-compatible endpoint.

After `.env` is configured, enable AI generation with `--use-ai`:

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --use-ai
```

Command-line options can temporarily override `.env` values:

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --use-ai \
  --model openai/custom-model \
  --ai-base-url https://llm.example.com/v1 \
  --ai-api-key sk-... \
  --ai-request-timeout 300 \
  --ai-transport-retries 1
```

`generation_config.json` records the final resolved `model`, per-attempt request timeout, and transport retry count, but does not record `OPENAI_BASE_URL` or `OPENAI_API_KEY`. When rerunning AI generation, provide connection settings again through `.env`, environment variables, or command-line options.

Each AI provider transport attempt times out after 300 seconds by default. Use `--ai-request-timeout` to select 1–600 seconds. Timeouts, connection failures, HTTP 429 responses, and major 5xx errors are retried at most once by default; use `--ai-transport-retries 0|1` to change this. LiteLLM internal retries are disabled so the project can record the elapsed time and exception category for every transport attempt. Authentication and request-parameter errors are not retried. Transport retries are independent from content repair: output that fails JSON, bar-count, note-length, character, or quality validation is repaired up to 2 times by default, configurable with `--ai-repair-retries`. Exhausting either boundary falls back to the rule-based generator.

Re-run a saved generation config:

```bash
tja-ai-chartgen generate-from-config output/generation_config.json
```

Start the local Web UI MVP:

```bash
tja-ai-chartgen web
tja-ai-chartgen web --host 0.0.0.0 --allow-remote
```

The Web UI supports audio upload, BPM/OFFSET/time-signature override, analysis preview, and rule-based or AI-enhanced regeneration of a selected bar range. Completed jobs open a playable chart preview with OGG playback, automatic note performance, timeline seeking, and don/ka hit sounds. Its collapsed AI options expose the per-attempt request timeout and 0–1 transport retries. Full-chart generation saves these non-sensitive settings for the result page, and local regenerate calls run the synchronous AI helper in a thread pool so they do not block the FastAPI event loop. It listens on `127.0.0.1:8000` by default and writes job files under `output/web/`. Listening on a non-loopback address requires the explicit `--allow-remote` option. This option only acknowledges the exposure risk; it does not add authentication or multi-user data isolation, so public deployments still require authentication and access control in front of the app. Each uploaded file is limited to 100 MiB. Remote mode disables exports to request-selected server directories, and job downloads expose only preview OGG files and generated TJA files, not AI sidecars or internal state files.

## Output

```txt
output/
├─ song.ogg
├─ song.tja
├─ analysis.json
├─ generation_config.json
├─ ai_input.json
├─ ai_attempts.json
├─ ai_output.json
├─ report.txt
└─ web/
   └─ <job-id>/
      ├─ <uploaded-audio>
      ├─ <stem>.ogg
      ├─ analysis.json
      ├─ chart_bars.json
      ├─ chart_options.json
      ├─ progress.json
      ├─ preview.tja
      ├─ result.html
      ├─ ai_input_<start>_<end>.json
      ├─ ai_attempts_<start>_<end>.json
      ├─ ai_output_<start>_<end>.json
      └─ regenerated_<start>_<end>.tja
```

## Reproducibility

Each `generate` run writes `generation_config.json` next to the TJA output. It records input path, metadata, difficulty, all-course mode, style, density, `--max-bars`, BPM/OFFSET overrides, time-signature override, BeatNet flag, special-note flag, AI flag, final resolved model name, content repair count, request timeout, and transport retry count. Use `generate-from-config` to run the same generation parameters again. `ai_attempts*.json` records content-validation attempts separately from actual transport attempts and includes a stable fallback reason when generation ultimately fails. API keys and base URLs are not written to `generation_config.json`; provide connection settings again through `.env`, environment variables, or command options when rerunning AI generation.

## Limitations

- Best for songs with stable BPM.
- Generated charts are drafts and require human review.
- OFFSET may need manual adjustment in OpenTaiko or another simulator.
- `--max-bars` is intended for quick draft checks and truncates the generated chart to the first N bars.
- `--bpm` and `--offset` override automatic analysis results for manual calibration.
- `--time-signature 4/4|3/4|6/8` overrides the analyzed meter and affects bar length, AI prompts, and `.tja` `#MEASURE` output.
- `--all-courses` writes `<stem>_easy.tja`, `<stem>_normal.tja`, `<stem>_hard.tja`, and `<stem>_oni.tja` with built-in level and density presets.
- `--density auto|low|medium|high|max` controls rule-based draft density and is passed into the AI prompt when `--use-ai` is enabled.
- `--style technical|stamina|hybrid|performance` selects the style template for rule patterns, special-note placement, and AI prompts.
- `--special-notes` allows simple drumroll and balloon notes; complex drumroll performances are still unsupported.
- `--use-beatnet` requires separately installing `BeatNet`; if BeatNet is unavailable or analysis fails, the CLI keeps the default librosa result.
- `--use-ai` can still fall back to the rule-based generator when the model is unavailable, output repair is exhausted, or credentials are misconfigured.
- The Web UI is a local MVP. It does not provide accounts, persistent task management, or a full chart editor.
- The MVP does not support BPM changes, branch charts, complex drumroll performances, or scroll gimmicks.

## Future Work

The following areas are still not implemented and are suitable for later versions:

- Support BPM changes, complex drumroll performances, branch charts, and scroll gimmicks for fuller TJA syntax coverage.
- Improve music structure analysis so generation is less dependent on stable-BPM songs.
- Expand the Web UI with task management, chart editing, and longer-term result storage.
- Build more reproducible reference-chart evaluation samples for comparing generation strategies.
