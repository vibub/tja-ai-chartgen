# tja-ai-chartgen

AI-assisted TJA chart draft generator for Taiko simulators.

Chinese documentation: [README.md](README.md).

License: MIT. See [LICENSE](LICENSE).

## Requirements

- Python 3.11+
- ffmpeg
- Optional Stage C: PyTorch, Demucs, and Transformers builds supported by the current Python/platform

## Install

Base development environment:

```bash
pip install -e ".[dev]"
```

Optional vocal and instrument analysis:

```bash
pip install -e ".[dev,instrument]"
tja-ai-chartgen prepare-instrument-models
```

The preparation command downloads pinned Demucs `htdemucs` and AST AudioSet weights to `models/instrument-v1/` under the project root by default. `models/` is ignored by Git. Generation never downloads models implicitly and never writes separated stems into the output directory.

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

Allowed `--density` values are `auto`, `low`, `medium`, `high`, and `max`. The rule-based fallback derives a duration-normalized load from course and level, while density adjusts within that difficulty range; `auto` also combines bar energy, transients, sustained activity, real phrase progress, and structural roles such as `build_up`, `peak`, and `breakdown`. Placements prioritize onset, strength, and accent features, fill sparse input from beat/downbeat grids, and use difficulty-specific notes/sec and occupancy caps for short high-BPM bars. Allowed `--style` values are `technical`, `stamina`, `hybrid`, and `performance`; they mainly affect rhythmic tendency, don/ka coloring, special-note cadence, and AI prompts.

Allow the rule generator and AI to use simple drumroll and balloon notes:

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --density high \
  --special-notes
```

When enabled, the rule generator only places single-bar `5...8` drumrolls and `7...8` balloons in active bars confirmed as `fill_candidate` by structure analysis. A `phrase_end`, `song_end`, or bar number divisible by 4/8 is not sufficient by itself. Start/end grids follow bar features, duration comes from the actual grid span and bar length, and balloon counts are derived from duration, course, and level. The writer emits the matching `BALLOON:` header. Quality reports record drumroll/balloon counts, duration, required balloon hits, and hits/sec separately; normal notes/sec excludes `5`, `7`, and `8` markers.

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

Audio features are first quantized to a canonical grid of 12 ticks per quarter note: 48 ticks per `4/4` bar and 36 ticks per `3/4` or `6/8` bar. A `ResolutionPlan` then uses reliable-onset quantization error to select 16/24/48 grids for `4/4`, or 12/18/36 grids for `3/4` and `6/8`. `song-global-v1` remains the stable default. When `structure-v1` finds complete, high-confidence phrases whose local rhythm genuinely needs finer timing, `phrase-stable-v2` upgrades the whole phrase only at phrase boundaries; if that would create too many switches, it falls back to one stable high resolution. Higher resolution only improves the representation of triplets, sixteenth notes, and mixed subdivisions; it does not automatically increase difficulty, which remains controlled by course, level, density, and musical evidence. Non-`4/4` bars are exported with `#MEASURE 3/4`.

`structure-v1` no longer cuts phrases on a fixed four-bar cycle. It combines energy percentiles, contextual changes, onset/activity density, 12-bin rhythm profiles, and `spectral-v1` semantics to produce variable-length `phrase_id` and `phrase_progress`, reusable `section_id` values, boundary confidence, `stable` / `build_up` / `peak` / `drop` / `cadence` / `breakdown` roles, and `fill_candidate_score`. These fields are persisted in `analysis.json` and sent to both the rule generator and the compact AI payload for gradual build-ups, peak contrast, breakdown skeletons, returning-section motifs, and evidence-based cadence fills.

`spectral-v1` uses HPSS to separate harmonic and percussive components, then extracts low/mid/high-band onset strength, spectral flux, brightness, harmonic novelty, texture novelty, and percussive ratio. Frame-level values are summarized into bar structure fields, while reliable attack positions are sent to the AI as sparse `spectral_events`. The rule fallback treats them as additional placement evidence, with low-frequency attacks acting as soft don evidence and high-frequency attacks as soft ka evidence. Short audio or a failed librosa spectral step does not stop the existing onset/RMS pipeline: generation continues with a `spectral-analysis-fallback` notice.

Optional `instrument-v1` uses local Demucs `htdemucs` separation for vocals, drums, bass, and other, then applies a pinned AST AudioSet classifier to the mix and other stem for stable guitar, piano/keyboard, strings, brass, woodwind, synth, and organ categories. Stem activity, vocal presence, dominant sources/instruments, and sparse vocal/drum/bass/accompaniment onsets are mapped to canonical bars and ticks. They strengthen phrase boundaries, build-ups, peaks, breakdowns, cadences, fills, AI motifs, and rule placements. These are always soft signals: vocals are not mapped syllable by syllable, and no label can override silence, difficulty, NPS, occupancy, resolution, or playability constraints.

Enable it explicitly:

```bash
tja-ai-chartgen generate song.mp3 \
  --title "Song Title" \
  --use-instrument-analysis \
  --instrument-device auto
```

`--instrument-device` accepts `auto`, `cpu`, `cuda`, or `mps`; `--instrument-model-dir` can point to a prepared local directory. Missing dependencies, models, devices, or model failures do not stop base generation. They produce structured partial/fallback notices instead.

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

Each AI provider transport attempt times out after 300 seconds by default. Use `--ai-request-timeout` to select 1–600 seconds. Timeouts, connection failures, HTTP 429 responses, and major 5xx errors are retried at most once by default; use `--ai-transport-retries 0|1` to change this. LiteLLM internal retries are disabled so the project can record the elapsed time and exception category for every transport attempt. Authentication and request-parameter errors are not retried. Transport retries are independent from content repair: the AI receives the `tja-ai-chartgen-compact-v2` payload and returns sparse `hits` / `long_notes` events on canonical ticks instead of choosing a notes-string length. The shared event encoder validates tick bounds, target-resolution representability, conflicts, and long-note structure. Legacy `notes` output, invalid JSON/bar counts, and quality-gate failures enter content repair, which runs up to 2 times by default and is configurable with `--ai-repair-retries`. Exhausting either boundary falls back to the rule-based generator.

At runtime, the AI prompt selects one anonymous continuous intro, peak, and cadence window for the requested course and nearest level. The checked-in static dataset currently contains 69 windows generated offline from 5 local sources and 23 courses. It stores only anonymous source IDs, course/level, per-bar resolution/measure/BPM/GOGO data, and sparse events—not titles, WAVE values, external absolute paths, or complete notes strings. Runtime generation never accesses the source reference directory; unreadable static windows fall back to the older evenly sampled examples.

Re-run a saved generation config:

```bash
tja-ai-chartgen generate-from-config output/generation_config.json
```

Start the local Web UI MVP:

```bash
tja-ai-chartgen web
tja-ai-chartgen web --host 0.0.0.0 --allow-remote
tja-ai-chartgen web --host 0.0.0.0 --allow-remote --allow-instrument-analysis
```

The Web UI supports audio upload, BPM/OFFSET/time-signature override, analysis preview, and rule-based or AI-enhanced regeneration of a selected bar range. Completed jobs open a playable chart preview with OGG playback, automatic note performance, timeline seeking, and don/ka hit sounds. Low-confidence analysis, BeatNet fallback, `spectral-v1` fallback, `instrument-v1` complete/partial/fallback state, base-resolution fallback, AI fallback, and final write errors are represented as structured `GenerationNotice` records, shown on both progress and result pages, and persisted to `generation_notices*.json`; the CLI writes the same class of sidecar and includes notices in `report.txt`. Its collapsed AI options expose the per-attempt request timeout and 0–1 transport retries. Full-chart generation saves these non-sensitive settings for the result page, and local regenerate calls run the synchronous AI helper in a thread pool so they do not block the FastAPI event loop. It listens on `127.0.0.1:8000` by default and writes job files under `output/web/`. Listening on a non-loopback address requires the explicit `--allow-remote` option. This option only acknowledges the exposure risk; it does not add authentication or multi-user data isolation, so public deployments still require authentication and access control in front of the app. Each uploaded file is limited to 100 MiB. Remote mode disables exports to request-selected server directories; job downloads expose only preview OGG files and generated TJA files, not AI sidecars or internal state files, and public progress/result responses hide server paths, connection URLs, and full provider response details. Remote mode disables Demucs/AST inference unless the server administrator also starts the app with `--allow-instrument-analysis`.

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
├─ generation_notices.json
├─ report.txt
└─ web/
   └─ <job-id>/
      ├─ <uploaded-audio>
      ├─ <stem>.ogg
      ├─ analysis.json
      ├─ chart_bars.json
      ├─ chart_options.json
      ├─ progress.json
      ├─ generation_notices.json
      ├─ preview.tja
      ├─ result.html
      ├─ ai_input_<start>_<end>.json
      ├─ ai_attempts_<start>_<end>.json
      ├─ ai_output_<start>_<end>.json
      ├─ generation_notices_<start>_<end>.json
      └─ regenerated_<start>_<end>.tja
```

## Reproducibility

Each `generate` run writes `generation_config.json` next to the TJA output. It records input path, metadata, difficulty, all-course mode, style, density, `--max-bars`, BPM/OFFSET overrides, time-signature override, BeatNet flag, Stage C flag/device and optional model directory, special-note flag, AI flag, final resolved model name, content repair count, request timeout, and transport retry count. Use `generate-from-config` to run the same generation parameters again. `ai_attempts*.json` records content-validation attempts separately from actual transport attempts and includes a stable fallback reason when generation ultimately fails. API keys and base URLs are not written to `generation_config.json`; provide connection settings again through `.env`, environment variables, or command options when rerunning AI generation.

## Quality Evaluation

The four rule-based course loads are initially calibrated against anonymous aggregate statistics from real four-course charts and are regression-tested through project-generated 120 BPM sparse, 180 BPM dense, adaptive 16/24/48-resolution, and stable→build-up→peak→drop structure WAV fixtures; the same pipeline also verifies `spectral-v1` envelope alignment, non-empty flux, bar/sparse-grid mapping, and graceful fallback. In addition to whole-chart average and peak-bar notes/sec, `quality_report*.json` records active-bar average notes/sec, longest note-stream count/duration, accent coverage, normalized pattern repetition, structure-density correlation, peak contrast, build-up slope agreement, cadence variation, fill-candidate precision, returning-section motif consistency, drum-onset coverage, bass/downbeat alignment, vocal-phrase response, instrument-transition response, confident Stage C coverage, instrument-supported fills, and resolution changes/quantization error. Ordinary hit count, NPS, streams, accents, and repetition are measured by real time or normalized bar position, so equivalent rhythms do not change when encoded at 16, 24, or 48 grids. Ordinary load counts only notes `1`–`4`; special-note load remains separate. New structure metrics are report-only and do not yet participate in a weighted score or AI repair gate.

Run `python tools/analyze_reference_dataset.py <reference-dir>` to analyze a user-provided directory of same-stem audio/TJA pairs offline. It emits anonymous aggregate metrics for multiple courses, BPM changes, measures, GOGO sections, and variable resolutions. Run `python tools/build_reference_windows.py <reference-dir> --output src/tja_ai_chartgen/ai/reference_windows.json` to rebuild continuous prompt windows offline. Runtime generation never reads that directory, and the repository does not store the reference audio, titles, raw notes, or absolute paths. Automated gates cover silence preservation, the four-course gradient, sparse-music restraint, high-BPM caps, dense Hard/Oni separation, structural roles, phrase-stable resolution, determinism, and structural preflight. Coloring, play feel, fill quality, and perceived star rating still require human playtesting and listening. See [docs/quality-evaluation.md](docs/quality-evaluation.md) for metric definitions, anonymous calibration ranges, reproduction steps, and the manual checklist.

## Limitations

- Best for songs with stable BPM.
- Generated charts are drafts and require human review.
- OFFSET may need manual adjustment in OpenTaiko or another simulator.
- `--max-bars` is intended for quick draft checks and truncates the generated chart to the first N bars.
- `--bpm` and `--offset` override automatic analysis results for manual calibration.
- `--time-signature 4/4|3/4|6/8` overrides the analyzed meter and affects bar length, AI prompts, and `.tja` `#MEASURE` output.
- `--all-courses` writes `<stem>_easy.tja`, `<stem>_normal.tja`, `<stem>_hard.tja`, and `<stem>_oni.tja` using built-in Easy 3, Normal 5, Hard 7, and Oni 10 levels to form a BPM-aware load gradient.
- `--density auto|low|medium|high|max` adjusts rule-based density within the selected course/level range; all four `--all-courses` charts share the user's density choice. Density is also passed into the AI prompt when `--use-ai` is enabled.
- `--style technical|stamina|hybrid|performance` selects rhythmic tendency, don/ka coloring, special-note candidate thresholds, and AI prompts; ordinary placements still prioritize analyzed music features.
- `--special-notes` adds single-bar drumrolls and duration-scaled balloons only at active fill candidates confirmed by structure analysis; cross-bar rolls, complex drumroll performances, and branch syntax are still unsupported.
- `--use-beatnet` requires separately installing `BeatNet`; if BeatNet is unavailable or analysis fails, the CLI keeps the default librosa result.
- `--use-instrument-analysis` requires optional dependencies and prepared local weights. Demucs and AST add substantial analysis time, memory use, and model storage. Source separation leakage and classifier errors remain possible, so all labels require human review.
- `--use-ai` can still fall back to the rule-based generator when the model is unavailable, output repair is exhausted, or credentials are misconfigured.
- The Web UI is a local MVP. It does not provide accounts, persistent task management, or a full chart editor.
- The MVP does not support BPM changes, branch charts, complex drumroll performances, or scroll gimmicks.

## Future Work

The following areas are still not implemented and are suitable for later versions:

- Support BPM changes, complex drumroll performances, branch charts, and scroll gimmicks for fuller TJA syntax coverage.
- Calibrate `spectral-v1`, `instrument-v1`, and structural thresholds with more real songs and playtesting, then decide which spectral/source/instrument/structure metrics should enter AI repair or CI gates.
- Expand the Web UI with task management, chart editing, and longer-term result storage.
