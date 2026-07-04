from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import ValidationError

from tja_ai_chartgen.audio.analyze import analyze_audio, apply_analysis_overrides
from tja_ai_chartgen.audio.convert import convert_to_ogg
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.features.meter import validate_time_signature
from tja_ai_chartgen.features.sections import assign_sections
from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars, validate_density
from tja_ai_chartgen.rules.styles import STYLE_LEVELS, validate_style
from tja_ai_chartgen.tja.model import ChartBar, ChartMetadata, SongAnalysis, TjaChart
from tja_ai_chartgen.tja.writer import render_tja
from tja_ai_chartgen.utils.paths import write_json

DEFAULT_WEB_OUTPUT_DIR = Path("output/web")
ALLOWED_WEB_NOTES = set("01234578")
DEFAULT_WEB_BALLOON_COUNT = 8


_PAGE_CSS = """
:root {
  color-scheme: dark;
  --bg: #101114;
  --bg-soft: #16181d;
  --surface: rgba(246, 240, 226, 0.08);
  --surface-strong: rgba(246, 240, 226, 0.13);
  --ink: #f6f0e2;
  --muted: #ada89a;
  --line: rgba(246, 240, 226, 0.16);
  --accent: #d6a85f;
  --accent-soft: rgba(214, 168, 95, 0.18);
  --danger: #e08a74;
  --shadow: 0 24px 80px rgba(4, 6, 10, 0.42);
  --radius-lg: 30px;
  --radius-md: 18px;
  --radius-sm: 12px;
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  min-height: 100dvh;
  margin: 0;
  color: var(--ink);
  font-family: "Outfit", "Aptos", "Segoe UI", "Noto Sans SC", sans-serif;
  line-height: 1.6;
  background:
    radial-gradient(circle at 14% -12%, rgba(214, 168, 95, 0.24), transparent 34rem),
    radial-gradient(circle at 88% 8%, rgba(119, 133, 117, 0.28), transparent 32rem),
    linear-gradient(135deg, #101114 0%, #17181d 48%, #12100e 100%);
  text-rendering: optimizeLegibility;
}

body::before {
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
  content: "";
  opacity: 0.16;
  background-image:
    linear-gradient(rgba(246, 240, 226, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(246, 240, 226, 0.04) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: radial-gradient(circle at top, black 0%, transparent 68%);
}

body::after {
  position: fixed;
  inset: 0;
  z-index: 10;
  pointer-events: none;
  content: "";
  opacity: 0.045;
  background-image: radial-gradient(rgba(255, 255, 255, 0.55) 0.5px, transparent 0.5px);
  background-size: 3px 3px;
}

.skip-link {
  position: fixed;
  top: 1rem;
  left: 1rem;
  z-index: 20;
  padding: 0.7rem 1rem;
  color: #16130e;
  background: var(--accent);
  border-radius: 999px;
  transform: translateY(-180%);
  transition: transform 180ms ease;
}

.skip-link:focus {
  transform: translateY(0);
}

.shell {
  width: min(1180px, calc(100% - 2rem));
  margin: 0 auto;
  padding: 1.25rem 0 4rem;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: clamp(2rem, 6vw, 5rem);
}

.brand {
  display: inline-flex;
  align-items: center;
  gap: 0.75rem;
  color: var(--ink);
  font-weight: 700;
  letter-spacing: -0.03em;
  text-decoration: none;
}

.brand-mark {
  display: grid;
  width: 2.4rem;
  height: 2.4rem;
  place-items: center;
  color: #17130d;
  background: var(--accent);
  border-radius: 0.8rem;
  box-shadow: 0 14px 34px rgba(214, 168, 95, 0.2);
}

.nav-note {
  color: var(--muted);
  font-size: 0.9rem;
  font-variant-numeric: tabular-nums;
}

.hero {
  display: grid;
  grid-template-columns: minmax(0, 0.9fr) minmax(22rem, 1.1fr);
  gap: clamp(1.2rem, 4vw, 3rem);
  align-items: start;
}

.hero-copy {
  position: sticky;
  top: 1.5rem;
  padding-top: 1.3rem;
}

.eyebrow,
.field-hint,
.meta-label,
th {
  color: var(--muted);
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

h1,
h2,
h3,
p {
  margin-top: 0;
}

h1 {
  max-width: 10ch;
  margin-bottom: 1.1rem;
  font-size: clamp(3.4rem, 10vw, 7.8rem);
  font-weight: 800;
  letter-spacing: -0.085em;
  line-height: 0.84;
  text-wrap: balance;
}

h2 {
  margin-bottom: 1rem;
  font-size: clamp(1.7rem, 3vw, 2.45rem);
  font-weight: 760;
  letter-spacing: -0.055em;
  line-height: 1;
}

.lede {
  max-width: 34rem;
  color: #d7d0c0;
  font-size: clamp(1.05rem, 1.7vw, 1.24rem);
  text-wrap: pretty;
}

.panel,
.metric-card,
.table-wrap,
.code-card,
.notice {
  background: linear-gradient(145deg, rgba(246, 240, 226, 0.11), rgba(246, 240, 226, 0.055));
  border: 1px solid var(--line);
  box-shadow: var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(22px);
}

.panel {
  position: relative;
  overflow: hidden;
  padding: clamp(1.1rem, 3vw, 2rem);
  border-radius: var(--radius-lg);
}

.panel::before {
  position: absolute;
  inset: 0 auto auto 10%;
  width: 65%;
  height: 1px;
  content: "";
  background: linear-gradient(90deg, transparent, rgba(214, 168, 95, 0.72), transparent);
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
  margin-top: 1.2rem;
}

.field {
  display: grid;
  gap: 0.45rem;
}

.field-wide {
  grid-column: 1 / -1;
}

label {
  color: var(--ink);
  font-weight: 650;
}

input,
select,
button,
textarea {
  font: inherit;
}

input,
select,
textarea {
  width: 100%;
  min-height: 3.15rem;
  padding: 0.78rem 0.9rem;
  color: var(--ink);
  background: rgba(12, 13, 16, 0.62);
  border: 1px solid rgba(246, 240, 226, 0.18);
  border-radius: var(--radius-sm);
  outline: none;
  transition:
    border-color 180ms ease,
    box-shadow 180ms ease,
    background 180ms ease;
}

input[type="file"] {
  padding: 0.72rem;
}

input[type="checkbox"] {
  width: 1.1rem;
  min-height: auto;
  accent-color: var(--accent);
}

input:focus,
select:focus,
button:focus-visible,
a:focus-visible {
  outline: 3px solid rgba(214, 168, 95, 0.36);
  outline-offset: 3px;
}

input:focus,
select:focus,
textarea:focus {
  background: rgba(12, 13, 16, 0.82);
  border-color: rgba(214, 168, 95, 0.78);
  box-shadow: 0 0 0 4px rgba(214, 168, 95, 0.11);
}

.checkbox-card {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  min-height: 3.15rem;
  padding: 0.78rem 0.9rem;
  background: rgba(12, 13, 16, 0.44);
  border: 1px solid rgba(246, 240, 226, 0.12);
  border-radius: var(--radius-sm);
}

button,
.button-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 3.2rem;
  padding: 0.8rem 1.15rem;
  color: #16130d;
  font-weight: 800;
  letter-spacing: -0.02em;
  text-decoration: none;
  cursor: pointer;
  background: linear-gradient(180deg, #e5bd76, #c99140);
  border: 0;
  border-radius: 999px;
  box-shadow: 0 16px 34px rgba(214, 168, 95, 0.22);
  transition:
    transform 180ms ease,
    box-shadow 180ms ease,
    filter 180ms ease;
}

button:hover,
.button-link:hover {
  filter: brightness(1.04);
  box-shadow: 0 20px 44px rgba(214, 168, 95, 0.3);
  transform: translateY(-1px);
}

button:active,
.button-link:active {
  transform: translateY(1px) scale(0.985);
}

form[data-loading="true"] button[type="submit"] {
  color: rgba(22, 19, 13, 0.72);
  cursor: progress;
}

form[data-loading="true"] button[type="submit"]::after {
  width: 0.85rem;
  height: 0.85rem;
  margin-left: 0.7rem;
  content: "";
  border: 2px solid rgba(22, 19, 13, 0.22);
  border-top-color: #16130d;
  border-radius: 999px;
  animation: spin 720ms linear infinite;
}

@keyframes spin {
  to { transform: rotate(1turn); }
}

.helper-strip,
.summary-head,
.result-path {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
}

.helper-strip {
  margin-top: 1rem;
  color: var(--muted);
  font-size: 0.92rem;
}

.badge {
  display: inline-flex;
  align-items: center;
  min-height: 1.8rem;
  padding: 0.25rem 0.58rem;
  color: #f8e7bf;
  background: rgba(214, 168, 95, 0.11);
  border: 1px solid rgba(214, 168, 95, 0.24);
  border-radius: 0.6rem;
}

.steps {
  display: grid;
  gap: 0.85rem;
  max-width: 30rem;
  padding: 0;
  margin: 2rem 0 0;
  list-style: none;
}

.steps li {
  display: grid;
  grid-template-columns: 2.25rem 1fr;
  gap: 0.85rem;
  align-items: start;
  color: #d7d0c0;
}

.step-no {
  display: grid;
  width: 2.25rem;
  height: 2.25rem;
  place-items: center;
  color: var(--accent);
  font-weight: 800;
  font-variant-numeric: tabular-nums;
  background: rgba(214, 168, 95, 0.1);
  border-radius: 0.75rem;
}

.stack {
  display: grid;
  gap: 1.25rem;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.85rem;
  margin: 1.2rem 0;
}

.metric-card {
  padding: 1rem;
  border-radius: var(--radius-md);
}

.metric-value {
  display: block;
  margin-top: 0.35rem;
  font-size: clamp(1.35rem, 3vw, 2rem);
  font-weight: 790;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.045em;
}

code,
pre,
td:first-child,
.metric-value {
  font-family: "Cascadia Mono", "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
}

code {
  padding: 0.18rem 0.38rem;
  color: #f6d99c;
  word-break: break-all;
  background: rgba(12, 13, 16, 0.66);
  border: 1px solid rgba(246, 240, 226, 0.1);
  border-radius: 0.45rem;
}

.table-wrap {
  overflow: hidden;
  border-radius: var(--radius-md);
}

.table-scroll {
  overflow-x: auto;
}

table {
  width: 100%;
  min-width: 42rem;
  border-collapse: collapse;
}

th,
td {
  padding: 0.82rem 0.9rem;
  text-align: left;
  border-bottom: 1px solid rgba(246, 240, 226, 0.09);
}

tbody tr:hover {
  background: rgba(214, 168, 95, 0.06);
}

td {
  color: #e7dfcf;
  font-variant-numeric: tabular-nums;
}

pre {
  max-height: 36rem;
  padding: 1.1rem;
  margin: 0;
  overflow: auto;
  color: #f4e8d0;
  background: rgba(5, 6, 8, 0.82);
  border-radius: var(--radius-md);
}

.code-card {
  display: grid;
  gap: 0.8rem;
  padding: 1rem;
  border-radius: var(--radius-lg);
}

.audio-card audio {
  width: 100%;
  margin-top: 0.8rem;
  accent-color: var(--accent);
}

.chart-editor {
  display: grid;
  gap: 1rem;
}

.editor-card {
  display: grid;
  gap: 1rem;
}

.editor-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
  justify-content: space-between;
}

.note-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  padding: 0;
  margin: 0;
  list-style: none;
}

.note-legend li {
  display: inline-flex;
  gap: 0.35rem;
  align-items: center;
  padding: 0.22rem 0.5rem;
  color: #ded4c0;
  font-size: 0.82rem;
  background: rgba(12, 13, 16, 0.44);
  border: 1px solid rgba(246, 240, 226, 0.1);
  border-radius: 999px;
}

.bar-editor {
  display: grid;
  grid-template-columns: 5.5rem minmax(18rem, 1fr) minmax(12rem, 0.55fr);
  gap: 0.8rem;
  align-items: start;
  padding: 0.9rem;
  background: rgba(12, 13, 16, 0.42);
  border: 1px solid rgba(246, 240, 226, 0.1);
  border-radius: var(--radius-md);
}

.bar-meta {
  display: grid;
  gap: 0.22rem;
  font-variant-numeric: tabular-nums;
}

.bar-no {
  color: #f8e7bf;
  font-weight: 820;
}

.bar-subtle {
  color: var(--muted);
  font-size: 0.82rem;
}

.note-grid {
  display: grid;
  grid-template-columns: repeat(var(--grid-count), minmax(1.9rem, 1fr));
  gap: 0.35rem;
  padding: 0.4rem;
  overflow-x: auto;
  background: rgba(5, 6, 8, 0.4);
  border: 1px solid rgba(246, 240, 226, 0.08);
  border-radius: var(--radius-sm);
}

.note-cell {
  display: grid;
  min-width: 1.9rem;
  min-height: 2.3rem;
  padding: 0;
  place-items: center;
  color: #f8edda;
  font-size: 0.72rem;
  font-weight: 900;
  letter-spacing: 0;
  background: rgba(246, 240, 226, 0.08);
  border: 1px solid rgba(246, 240, 226, 0.12);
  border-radius: 0.65rem;
  box-shadow: none;
}

.note-cell:hover {
  filter: brightness(1.1);
  box-shadow: none;
  transform: translateY(-1px);
}

.note-cell[data-beat="true"] {
  border-color: rgba(214, 168, 95, 0.5);
}

.note-0 { color: rgba(246, 240, 226, 0.42); background: rgba(246, 240, 226, 0.04); }
.note-1 { background: #d94f3f; }
.note-2 { background: #4f7fd9; }
.note-3 { background: linear-gradient(135deg, #d94f3f 0 50%, #f3c267 50%); }
.note-4 { background: linear-gradient(135deg, #4f7fd9 0 50%, #f3c267 50%); }
.note-5 { color: #17130d; background: #f3c267; }
.note-7 { color: #17130d; background: #8ed16f; }
.note-8 { color: #17130d; background: #d9d9d9; }

.note-text-grid {
  display: grid;
  gap: 0.55rem;
}

.note-text-grid input {
  min-height: 2.5rem;
  font-family: "Cascadia Mono", "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
  font-size: 0.9rem;
}

.note-text-grid label {
  display: grid;
  gap: 0.35rem;
}

.notice {
  padding: 1rem 1.1rem;
  border-radius: var(--radius-md);
}

.error {
  color: #ffe1d7;
  background: rgba(224, 138, 116, 0.12);
  border-color: rgba(224, 138, 116, 0.36);
}

.footer {
  margin-top: 4rem;
  color: var(--muted);
  font-size: 0.9rem;
}

@media (max-width: 860px) {
  .hero,
  .summary-grid {
    grid-template-columns: 1fr;
  }

  .hero-copy {
    position: static;
  }

  .form-grid,
  .bar-editor {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 560px) {
  .shell {
    width: min(100% - 1rem, 1180px);
    padding-top: 0.75rem;
  }

  .topbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .panel {
    border-radius: 22px;
  }
}
"""

_PAGE_SCRIPT = """
const NOTE_SEQUENCE = ['0', '1', '2', '3', '4', '5', '7', '8'];
const NOTE_LABELS = {
  '0': '·',
  '1': 'ド',
  '2': 'カ',
  '3': '大ド',
  '4': '大カ',
  '5': '連',
  '7': '風',
  '8': '止',
};

function paintNoteCell(cell, note) {
  const normalized = NOTE_SEQUENCE.includes(note) ? note : '0';
  const beatClass = cell.dataset.beat === 'true' ? ' beat-cell' : '';
  cell.dataset.note = normalized;
  cell.className = `note-cell note-${normalized}${beatClass}`;
  cell.textContent = NOTE_LABELS[normalized];
  cell.setAttribute('aria-label', `note ${normalized}`);
}

function syncBarNotes(row, notes) {
  const hidden = row.querySelector('input[type="hidden"][data-role="notes"]');
  const text = row.querySelector('input[data-role="notes-text"]');
  const cells = [...row.querySelectorAll('.note-cell')];
  const normalized = cells.map((cell, index) => NOTE_SEQUENCE.includes(notes[index]) ? notes[index] : '0');
  cells.forEach((cell, index) => paintNoteCell(cell, normalized[index] || '0'));
  const value = normalized.join('');
  if (hidden) hidden.value = value;
  if (text && text.value !== value) text.value = value;
}

function setupChartEditor() {
  document.querySelectorAll('.bar-editor').forEach((row) => {
    const cells = [...row.querySelectorAll('.note-cell')];
    const text = row.querySelector('input[data-role="notes-text"]');
    syncBarNotes(row, cells.map((cell) => cell.dataset.note || '0').join(''));

    cells.forEach((cell, index) => {
      cell.addEventListener('click', () => {
        const current = cell.dataset.note || '0';
        const next = NOTE_SEQUENCE[(NOTE_SEQUENCE.indexOf(current) + 1) % NOTE_SEQUENCE.length];
        const notes = cells.map((item, itemIndex) => itemIndex === index ? next : (item.dataset.note || '0')).join('');
        syncBarNotes(row, notes);
      });
    });

    if (text) {
      text.addEventListener('input', () => syncBarNotes(row, text.value));
    }
  });
}

setupChartEditor();

document.querySelectorAll('form').forEach((form) => {
  form.addEventListener('submit', () => {
    const button = form.querySelector('button[type="submit"]');
    form.dataset.loading = 'true';
    form.setAttribute('aria-busy', 'true');
    if (button && button.dataset.loadingText) {
      button.textContent = button.dataset.loadingText;
    }
  });
});
"""


def create_app(output_dir: Path = DEFAULT_WEB_OUTPUT_DIR) -> FastAPI:
    app = FastAPI(title="tja-ai-chartgen Web UI")
    app.state.output_dir = output_dir

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _page("tja-ai-chartgen", _analysis_form())

    @app.post("/analyze", response_class=HTMLResponse)
    async def analyze(
        audio: Annotated[UploadFile, File()],
        title: Annotated[str, Form()],
        artist: Annotated[str, Form()] = "",
        max_bars: Annotated[int | None, Form()] = None,
        bpm: Annotated[float | None, Form()] = None,
        offset: Annotated[float | None, Form()] = None,
        time_signature: Annotated[str, Form()] = "",
        use_beatnet: Annotated[bool, Form()] = False,
    ) -> HTMLResponse:
        try:
            job_dir = _new_job_dir(app.state.output_dir)
            input_path = _save_upload(job_dir, audio)
            ogg_path = job_dir / f"{input_path.stem}.ogg"
            convert_to_ogg(input_path, ogg_path)

            raw = analyze_audio(ogg_path, use_beatnet=use_beatnet)
            raw = apply_analysis_overrides(raw, bpm=bpm, offset=offset)
            if time_signature:
                validate_time_signature(time_signature)
                raw = raw.model_copy(update={"time_signature": time_signature})
            bars = assign_sections(build_bar_features(raw, max_bars=max_bars))
            analysis = SongAnalysis(
                title=title,
                artist=artist or None,
                audio_file=str(input_path),
                ogg_file=str(ogg_path),
                bpm=raw.bpm,
                offset=raw.offset,
                time_signature=raw.time_signature,
                bars=bars,
            )
            analysis_path = write_json(job_dir / "analysis.json", analysis)
            return HTMLResponse(
                _page(
                    "Analysis preview",
                    _analysis_summary(analysis, analysis_path, job_dir.name)
                    + _regenerate_form(job_dir.name),
                )
            )
        except Exception as error:  # noqa: BLE001 - Web boundary returns a readable error page.
            return HTMLResponse(
                _page("Analysis failed", _error_notice(str(error))),
                status_code=400,
            )

    @app.get("/jobs/{job_id}/{filename}")
    async def job_file(job_id: str, filename: str) -> FileResponse:
        try:
            job_dir = _job_dir(app.state.output_dir, job_id)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        path = job_dir / Path(filename).name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Job file not found: {filename}")
        return FileResponse(path)

    @app.post("/regenerate", response_class=HTMLResponse)
    async def regenerate(
        job_id: Annotated[str, Form()],
        start_bar: Annotated[int, Form()],
        end_bar: Annotated[int, Form()],
        course: Annotated[str, Form()] = "Oni",
        level: Annotated[int, Form()] = 10,
        style: Annotated[str, Form()] = "technical",
        density: Annotated[str, Form()] = "auto",
        special_notes: Annotated[bool, Form()] = False,
    ) -> HTMLResponse:
        try:
            validate_density(density)
            validate_style(style)
            job_dir = _job_dir(app.state.output_dir, job_id)
            analysis = SongAnalysis.model_validate_json(
                (job_dir / "analysis.json").read_text(encoding="utf-8")
            )
            selected_bars = _select_bars(analysis, start_bar, end_bar)
            chart_bars = generate_fallback_chart_bars(
                selected_bars,
                style=style,
                density=density,
                special_notes=special_notes,
            )
            chart = TjaChart(
                metadata=ChartMetadata(
                    title=analysis.title,
                    artist=analysis.artist,
                    wave=Path(analysis.ogg_file).name,
                    bpm=analysis.bpm,
                    offset=analysis.offset,
                    course=course,
                    level=level,
                ),
                bars=chart_bars,
            )
            tja_text = render_tja(chart)
            output_path = job_dir / f"regenerated_{start_bar}_{end_bar}.tja"
            output_path.write_text(tja_text, encoding="utf-8")
            body = "".join(
                [
                    _result_panel(
                        job_id=job_id,
                        output_path=output_path,
                        tja_text=tja_text,
                        analysis=analysis,
                        chart_bars=chart_bars,
                        course=course,
                        level=level,
                    ),
                    _regenerate_form(job_id),
                ]
            )
            return HTMLResponse(_page("Regenerated bars", body))
        except (FileNotFoundError, ValidationError, ValueError) as error:
            return HTMLResponse(
                _page("Regeneration failed", _error_notice(str(error))),
                status_code=400,
            )

    @app.post("/save-chart", response_class=HTMLResponse)
    async def save_chart(request: Request) -> HTMLResponse:
        try:
            form = await request.form()
            job_id = str(form.get("job_id", ""))
            job_dir = _job_dir(app.state.output_dir, job_id)
            analysis = SongAnalysis.model_validate_json(
                (job_dir / "analysis.json").read_text(encoding="utf-8")
            )
            course = str(form.get("course", "Oni"))
            level = int(str(form.get("level", "10")))
            chart_bars = _edited_chart_bars_from_form(form)
            chart = TjaChart(
                metadata=ChartMetadata(
                    title=analysis.title,
                    artist=analysis.artist,
                    wave=Path(analysis.ogg_file).name,
                    bpm=analysis.bpm,
                    offset=analysis.offset,
                    course=course,
                    level=level,
                ),
                bars=chart_bars,
            )
            tja_text = render_tja(chart)
            output_path = job_dir / _edited_output_filename(chart_bars)
            output_path.write_text(tja_text, encoding="utf-8")
            body = "".join(
                [
                    _result_panel(
                        job_id=job_id,
                        output_path=output_path,
                        tja_text=tja_text,
                        analysis=analysis,
                        chart_bars=chart_bars,
                        course=course,
                        level=level,
                    ),
                    _regenerate_form(job_id),
                ]
            )
            return HTMLResponse(_page("Saved chart edits", body))
        except (FileNotFoundError, ValidationError, ValueError) as error:
            return HTMLResponse(
                _page("Save failed", _error_notice(str(error))),
                status_code=400,
            )

    return app


app = create_app()


def _new_job_dir(output_dir: Path) -> Path:
    job_dir = output_dir / uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


def _job_dir(output_dir: Path, job_id: str) -> Path:
    if not job_id.isalnum():
        raise ValueError("Invalid job id")
    job_dir = output_dir / job_id
    if not job_dir.exists():
        raise FileNotFoundError(f"Job not found: {job_id}")
    return job_dir


def _save_upload(job_dir: Path, audio: UploadFile) -> Path:
    filename = Path(audio.filename or "upload.audio").name
    path = job_dir / filename
    content = audio.file.read()
    if not content:
        raise ValueError("Uploaded audio is empty")
    path.write_bytes(content)
    return path


def _select_bars(analysis: SongAnalysis, start_bar: int, end_bar: int):
    if start_bar < 1:
        raise ValueError("start_bar must be greater than or equal to 1")
    if end_bar < start_bar:
        raise ValueError("end_bar must be greater than or equal to start_bar")
    selected = [bar for bar in analysis.bars if start_bar - 1 <= bar.index <= end_bar - 1]
    if not selected:
        raise ValueError("No bars selected")
    return selected


def _edited_chart_bars_from_form(form) -> list[ChartBar]:
    bar_count = int(str(form.get("bar_count", "0")))
    if bar_count < 1:
        raise ValueError("No editable bars submitted")

    chart_bars: list[ChartBar] = []
    for position in range(bar_count):
        notes = str(form.get(f"notes_{position}", ""))
        expected_length = int(str(form.get(f"grids_per_bar_{position}", len(notes))))
        if len(notes) != expected_length:
            raise ValueError(
                f"notes_{position} must contain exactly {expected_length} character(s), got {len(notes)}"
            )
        illegal_characters = sorted(set(notes) - ALLOWED_WEB_NOTES)
        if illegal_characters:
            raise ValueError(f"notes_{position} contains illegal character(s): {''.join(illegal_characters)}")

        time_signature = str(form.get(f"time_signature_{position}", "4/4"))
        validate_time_signature(time_signature)
        chart_bars.append(
            ChartBar(
                index=int(str(form.get(f"bar_index_{position}", position))),
                notes=notes,
                time_signature=time_signature,
                balloon_counts=_parse_balloon_counts(
                    str(form.get(f"balloon_counts_{position}", "")),
                    notes.count("7"),
                ),
            )
        )

    return chart_bars


def _parse_balloon_counts(raw_value: str, expected_count: int) -> list[int]:
    if expected_count == 0:
        return []
    if not raw_value.strip():
        return [DEFAULT_WEB_BALLOON_COUNT] * expected_count

    counts: list[int] = []
    for raw_count in raw_value.split(","):
        value = int(raw_count.strip())
        if value < 1:
            raise ValueError("balloon counts must be positive integers")
        counts.append(value)

    if len(counts) != expected_count:
        raise ValueError(f"balloon counts must contain exactly {expected_count} value(s)")
    return counts


def _edited_output_filename(chart_bars: list[ChartBar]) -> str:
    first_bar = chart_bars[0].index + 1
    last_bar = chart_bars[-1].index + 1
    return f"edited_{first_bar}_{last_bar}.tja"


def _analysis_form() -> str:
    return f"""
<section class="hero" aria-labelledby="page-title">
  <div class="hero-copy">
    <p class="eyebrow">Local chart workbench</p>
    <h1 id="page-title">tja-ai-chartgen</h1>
    <p class="lede">
      上传音频，先做节拍与小节分析，再挑选片段生成可检查的 TJA 草稿。
      这个界面适合快速试谱、修正 BPM/OFFSET，以及比较不同风格模板。
    </p>
    <ol class="steps" aria-label="生成流程">
      <li><span class="step-no">01</span><span>导入音频并转换为项目使用的 OGG 文件。</span></li>
      <li><span class="step-no">02</span><span>提取 BPM、OFFSET、拍号、小节能量和段落标签。</span></li>
      <li><span class="step-no">03</span><span>选择小节范围，重新生成局部 TJA 片段。</span></li>
    </ol>
  </div>

  <section class="panel" aria-labelledby="upload-heading">
    <p class="eyebrow">Analyze audio</p>
    <h2 id="upload-heading">Upload and analyze</h2>
    <form action="/analyze" enctype="multipart/form-data" method="post">
      <div class="form-grid">
        <label class="field field-wide">
          Audio
          <input name="audio" type="file" accept="audio/*" required>
          <span class="field-hint">mp3、wav、flac 等 ffmpeg 可读取的音频。</span>
        </label>
        <label class="field">
          Title
          <input name="title" placeholder="Song Title" required>
        </label>
        <label class="field">
          Artist
          <input name="artist" placeholder="可选">
        </label>
        <label class="field">
          Max bars
          <input name="max_bars" type="number" min="1" placeholder="16">
        </label>
        <label class="field">
          BPM override
          <input name="bpm" type="number" step="0.001" min="0" placeholder="220.588">
        </label>
        <label class="field">
          OFFSET override
          <input name="offset" type="number" step="0.001" placeholder="0.725">
        </label>
        <label class="field">
          Time signature
          <select name="time_signature">
            <option value="">使用分析结果</option>
            <option value="4/4">4/4</option>
            <option value="3/4">3/4</option>
            <option value="6/8">6/8</option>
          </select>
        </label>
        <label class="checkbox-card field-wide">
          <input name="use_beatnet" type="checkbox" value="true">
          <span>Use BeatNet <span class="field-hint">尝试增强 downbeat、meter 和 offset。</span></span>
        </label>
      </div>
      <div class="helper-strip">
        <button type="submit" data-loading-text="Analyzing">Upload and analyze</button>
        <span>Style options: {', '.join(STYLE_LEVELS)}</span>
      </div>
    </form>
  </section>
</section>
"""


def _analysis_summary(analysis: SongAnalysis, analysis_path: Path, job_id: str) -> str:
    rows = "".join(
        f"<tr><td>{bar.index + 1}</td><td>{bar.start_time:.3f}</td><td>{bar.end_time:.3f}</td>"
        f"<td>{bar.energy:.3f}</td><td>{_escape(bar.section)}</td><td>{bar.grids_per_bar}</td></tr>"
        for bar in analysis.bars
    )
    return f"""
<section class="stack" aria-labelledby="analysis-heading">
  <div class="summary-head">
    <span class="badge">Job <code>{_escape(job_id)}</code></span>
    <span class="badge">Analysis JSON <code>{_escape(str(analysis_path))}</code></span>
  </div>
  <section class="panel">
    <p class="eyebrow">Analysis preview</p>
    <h1 id="analysis-heading">Analysis preview</h1>
    <p class="lede">分析结果已保存。确认 BPM、OFFSET 和小节数量后，可以只重生成需要调整的片段。</p>
    <div class="summary-grid" aria-label="分析指标">
      <article class="metric-card">
        <span class="meta-label">BPM: {analysis.bpm}</span>
        <span class="metric-value">{analysis.bpm:.3f}</span>
      </article>
      <article class="metric-card">
        <span class="meta-label">OFFSET: {analysis.offset}</span>
        <span class="metric-value">{analysis.offset:.3f}</span>
      </article>
      <article class="metric-card">
        <span class="meta-label">Time signature</span>
        <span class="metric-value">{_escape(analysis.time_signature)}</span>
      </article>
      <article class="metric-card">
        <span class="meta-label">Bars</span>
        <span class="metric-value">{len(analysis.bars)}</span>
      </article>
    </div>
  </section>

  {_audio_preview(analysis, job_id)}

  <section class="table-wrap" aria-labelledby="bars-heading">
    <div class="panel">
      <p class="eyebrow">Bar map</p>
      <h2 id="bars-heading">小节预览</h2>
    </div>
    <div class="table-scroll">
      <table>
        <thead>
          <tr><th>Bar</th><th>Start</th><th>End</th><th>Energy</th><th>Section</th><th>Grids</th></tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </section>
</section>
"""


def _regenerate_form(job_id: str) -> str:
    return f"""
<section class="panel" aria-labelledby="regen-heading">
  <p class="eyebrow">Pattern draft</p>
  <h2 id="regen-heading">Regenerate selected bars</h2>
  <p class="lede">选择起止小节、难度和模板，只输出这段片段，方便你逐步修正谱面。</p>
  <form action="/regenerate" method="post">
    <input name="job_id" type="hidden" value="{_escape(job_id)}">
    <div class="form-grid">
      <label class="field">
        Start bar
        <input name="start_bar" type="number" min="1" value="1" required>
      </label>
      <label class="field">
        End bar
        <input name="end_bar" type="number" min="1" value="1" required>
      </label>
      <label class="field">
        Course
        <input name="course" value="Oni">
      </label>
      <label class="field">
        Level
        <input name="level" type="number" min="1" max="10" value="10">
      </label>
      <label class="field">
        Style
        <select name="style">{_option_tags(STYLE_LEVELS, "technical")}</select>
      </label>
      <label class="field">
        Density
        <select name="density">{_option_tags(("auto", "low", "medium", "high", "max"), "auto")}</select>
      </label>
      <label class="checkbox-card field-wide">
        <input name="special_notes" type="checkbox" value="true">
        <span>Special notes <span class="field-hint">允许简单滚奏和气球。</span></span>
      </label>
    </div>
    <div class="helper-strip">
      <button type="submit" data-loading-text="Regenerating">Regenerate</button>
      <span>当前 job：<code>{_escape(job_id)}</code></span>
    </div>
  </form>
</section>
"""


def _audio_preview(analysis: SongAnalysis, job_id: str) -> str:
    ogg_name = Path(analysis.ogg_file).name
    return f"""
  <section class="panel audio-card" aria-labelledby="audio-heading">
    <p class="eyebrow">Audio preview</p>
    <h2 id="audio-heading">音频预览</h2>
    <p class="lede">播放转换后的 OGG，配合下方小节起止时间检查 OFFSET 与局部节奏。</p>
    <audio controls preload="metadata" src="/jobs/{_escape(job_id)}/{_escape(ogg_name)}"></audio>
  </section>
"""


def _chart_editor(
    job_id: str,
    analysis: SongAnalysis,
    chart_bars: list[ChartBar],
    course: str,
    level: int,
) -> str:
    rows = "".join(
        _bar_editor_row(position, bar, analysis)
        for position, bar in enumerate(chart_bars)
    )
    return f"""
  <section class="panel chart-editor" aria-labelledby="chart-editor-heading">
    <div class="editor-toolbar">
      <div>
        <p class="eyebrow">Interactive preview</p>
        <h2 id="chart-editor-heading">谱面预览与调整</h2>
      </div>
      {_note_legend()}
    </div>
    <form class="editor-card" action="/save-chart" method="post">
      <input name="job_id" type="hidden" value="{_escape(job_id)}">
      <input name="course" type="hidden" value="{_escape(course)}">
      <input name="level" type="hidden" value="{level}">
      <input name="bar_count" type="hidden" value="{len(chart_bars)}">
      {rows}
      <div class="helper-strip">
        <button type="submit" data-loading-text="Saving">保存调整后的 TJA</button>
        <span>点击格子循环 0/1/2/3/4/5/7/8；文本框可直接粘贴 TJA 小节音符。</span>
      </div>
    </form>
  </section>
"""


def _bar_editor_row(position: int, chart_bar: ChartBar, analysis: SongAnalysis) -> str:
    feature = next((bar for bar in analysis.bars if bar.index == chart_bar.index), None)
    beat_grids = set(feature.beat_grids if feature else [])
    cells = "".join(
        _note_cell(note, grid_index in beat_grids)
        for grid_index, note in enumerate(chart_bar.notes)
    )
    balloon_counts = ",".join(str(count) for count in chart_bar.balloon_counts)
    start_time = f"{feature.start_time:.3f}s" if feature else "unknown"
    section = feature.section if feature else "unknown"
    return f"""
      <article class="bar-editor" data-bar-index="{chart_bar.index}">
        <div class="bar-meta">
          <span class="bar-no">Bar {chart_bar.index + 1}</span>
          <span class="bar-subtle">{_escape(start_time)}</span>
          <span class="bar-subtle">{_escape(section)} · {chart_bar.time_signature}</span>
        </div>
        <div class="note-grid" style="--grid-count: {len(chart_bar.notes)}" aria-label="Bar {chart_bar.index + 1} note grid">
          {cells}
        </div>
        <div class="note-text-grid">
          <input name="bar_index_{position}" type="hidden" value="{chart_bar.index}">
          <input name="time_signature_{position}" type="hidden" value="{_escape(chart_bar.time_signature)}">
          <input name="grids_per_bar_{position}" type="hidden" value="{len(chart_bar.notes)}">
          <input name="notes_{position}" type="hidden" data-role="notes" value="{_escape(chart_bar.notes)}">
          <label>
            Notes
            <input data-role="notes-text" value="{_escape(chart_bar.notes)}" maxlength="{len(chart_bar.notes)}">
          </label>
          <label>
            Balloon counts
            <input name="balloon_counts_{position}" value="{_escape(balloon_counts)}" placeholder="8,8">
          </label>
        </div>
      </article>
"""


def _note_cell(note: str, is_beat: bool) -> str:
    normalized = note if note in ALLOWED_WEB_NOTES else "0"
    label = {
        "0": "·",
        "1": "ド",
        "2": "カ",
        "3": "大ド",
        "4": "大カ",
        "5": "連",
        "7": "風",
        "8": "止",
    }[normalized]
    beat_value = "true" if is_beat else "false"
    return f"""
          <button class="note-cell note-{normalized}" type="button" data-note="{normalized}" data-beat="{beat_value}" aria-label="note {normalized}">{label}</button>
"""


def _note_legend() -> str:
    items = "".join(
        f'<li><span class="note-cell note-{note}" aria-hidden="true">{label}</span><span>{note}</span></li>'
        for note, label in (
            ("0", "·"),
            ("1", "ド"),
            ("2", "カ"),
            ("3", "大ド"),
            ("4", "大カ"),
            ("5", "連"),
            ("7", "風"),
            ("8", "止"),
        )
    )
    return f'<ul class="note-legend" aria-label="音符图例">{items}</ul>'


def _result_panel(
    *,
    job_id: str,
    output_path: Path,
    tja_text: str,
    analysis: SongAnalysis,
    chart_bars: list[ChartBar],
    course: str,
    level: int,
) -> str:
    return f"""
<section class="stack" aria-labelledby="result-heading">
  <section class="panel">
    <p class="eyebrow">Export ready</p>
    <h1 id="result-heading">Regenerated bars</h1>
    <p class="result-path">Generated: <code>{_escape(str(output_path))}</code></p>
    <p class="lede">参考 PeepoDrumKit 这类太鼓谱面编辑器，把 TJA 文本变成可点选的鼓点网格；点击格子循环音符，再保存为新的 .tja。</p>
  </section>
  {_chart_editor(job_id, analysis, chart_bars, course, level)}
  <section class="code-card" aria-label="TJA preview">
    <p class="eyebrow">TJA preview</p>
    <pre>{_escape(tja_text)}</pre>
  </section>
</section>
"""


def _error_notice(message: str) -> str:
    return f"""
<section class="panel" aria-labelledby="error-heading">
  <p class="eyebrow">Request failed</p>
  <h1 id="error-heading">无法完成操作</h1>
  <p class="notice error">{_escape(message)}</p>
  <a class="button-link" href="/">返回上传页面</a>
</section>
"""


def _option_tags(options, selected: str) -> str:
    return "".join(
        f'<option value="{_escape(option)}"{_selected_attr(option, selected)}>{_escape(option)}</option>'
        for option in options
    )


def _selected_attr(option: str, selected: str) -> str:
    return " selected" if option == selected else ""


def _page(title: str, body: str) -> str:
    return f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="tja-ai-chartgen 本地 Web UI，用于分析音频并生成 TJA 谱面草稿。">
  <meta property="og:title" content="{_escape(title)}">
  <meta property="og:description" content="本地音频分析和 TJA 谱面草稿工作台。">
  <meta name="theme-color" content="#101114">
  <title>{_escape(title)}</title>
  <style>{_PAGE_CSS}</style>
</head>
<body>
  <a class="skip-link" href="#main">跳到主要内容</a>
  <div class="shell">
    <header class="topbar">
      <a class="brand" href="/" aria-label="tja-ai-chartgen 首页">
        <span class="brand-mark" aria-hidden="true">太</span>
        <span>tja-ai-chartgen</span>
      </a>
      <span class="nav-note">local web ui · no cloud upload</span>
    </header>
    <main id="main">
      {body}
    </main>
    <footer class="footer">
      <p>本地工具界面。输出文件保存在配置的 <code>output/web</code> job 目录中。</p>
    </footer>
  </div>
  <script>{_PAGE_SCRIPT}</script>
</body>
</html>
"""


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
