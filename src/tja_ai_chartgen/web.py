from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from tja_ai_chartgen.audio.analyze import analyze_audio, apply_analysis_overrides
from tja_ai_chartgen.audio.convert import convert_to_ogg
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.features.meter import validate_time_signature
from tja_ai_chartgen.features.sections import assign_sections
from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars, validate_density
from tja_ai_chartgen.rules.styles import STYLE_LEVELS, validate_style
from tja_ai_chartgen.tja.model import ChartMetadata, SongAnalysis, TjaChart
from tja_ai_chartgen.tja.writer import render_tja
from tja_ai_chartgen.utils.paths import write_json

DEFAULT_WEB_OUTPUT_DIR = Path("output/web")


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
                    _analysis_summary(analysis, analysis_path, job_dir.name) + _regenerate_form(job_dir.name),
                )
            )
        except Exception as error:  # noqa: BLE001 - Web boundary returns a readable error page.
            return HTMLResponse(_page("Analysis failed", f"<p class='error'>{_escape(str(error))}</p>"), status_code=400)

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
            analysis = SongAnalysis.model_validate_json((job_dir / "analysis.json").read_text(encoding="utf-8"))
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
                    f"<p>Generated: <code>{_escape(str(output_path))}</code></p>",
                    "<pre>",
                    _escape(tja_text),
                    "</pre>",
                    _regenerate_form(job_id),
                ]
            )
            return HTMLResponse(_page("Regenerated bars", body))
        except (FileNotFoundError, ValidationError, ValueError) as error:
            return HTMLResponse(_page("Regeneration failed", f"<p class='error'>{_escape(str(error))}</p>"), status_code=400)

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


def _analysis_form() -> str:
    return f"""
<form action="/analyze" enctype="multipart/form-data" method="post">
  <label>Audio <input name="audio" type="file" required></label>
  <label>Title <input name="title" required></label>
  <label>Artist <input name="artist"></label>
  <label>Max bars <input name="max_bars" type="number" min="1"></label>
  <label>BPM override <input name="bpm" type="number" step="0.001" min="0"></label>
  <label>OFFSET override <input name="offset" type="number" step="0.001"></label>
  <label>Time signature <input name="time_signature" placeholder="4/4, 3/4, 6/8"></label>
  <label><input name="use_beatnet" type="checkbox" value="true"> Use BeatNet</label>
  <button type="submit">Upload and analyze</button>
</form>
<p>Style options: {', '.join(STYLE_LEVELS)}</p>
"""


def _analysis_summary(analysis: SongAnalysis, analysis_path: Path, job_id: str) -> str:
    rows = "".join(
        f"<tr><td>{bar.index + 1}</td><td>{bar.start_time:.3f}</td><td>{bar.end_time:.3f}</td>"
        f"<td>{bar.energy:.3f}</td><td>{_escape(bar.section)}</td><td>{bar.grids_per_bar}</td></tr>"
        for bar in analysis.bars
    )
    return f"""
<p>Job: <code>{_escape(job_id)}</code></p>
<p>Analysis JSON: <code>{_escape(str(analysis_path))}</code></p>
<ul>
  <li>BPM: {analysis.bpm}</li>
  <li>OFFSET: {analysis.offset}</li>
  <li>Time signature: {_escape(analysis.time_signature)}</li>
  <li>Bars: {len(analysis.bars)}</li>
</ul>
<table>
  <thead><tr><th>Bar</th><th>Start</th><th>End</th><th>Energy</th><th>Section</th><th>Grids</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


def _regenerate_form(job_id: str) -> str:
    return f"""
<h2>Regenerate selected bars</h2>
<form action="/regenerate" method="post">
  <input name="job_id" type="hidden" value="{_escape(job_id)}">
  <label>Start bar <input name="start_bar" type="number" min="1" value="1" required></label>
  <label>End bar <input name="end_bar" type="number" min="1" value="1" required></label>
  <label>Course <input name="course" value="Oni"></label>
  <label>Level <input name="level" type="number" min="1" max="10" value="10"></label>
  <label>Style <input name="style" value="technical"></label>
  <label>Density <input name="density" value="auto"></label>
  <label><input name="special_notes" type="checkbox" value="true"> Special notes</label>
  <button type="submit">Regenerate</button>
</form>
"""


def _page(title: str, body: str) -> str:
    return f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{_escape(title)}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; line-height: 1.5; }}
    form {{ display: grid; gap: .75rem; max-width: 42rem; margin-bottom: 2rem; }}
    label {{ display: grid; gap: .25rem; }}
    table {{ border-collapse: collapse; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ccc; padding: .35rem .6rem; }}
    pre {{ background: #111; color: #eee; padding: 1rem; overflow: auto; }}
    .error {{ color: #b00020; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>{_escape(title)}</h1>
  {body}
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
