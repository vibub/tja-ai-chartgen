from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console

from tja_ai_chartgen import __version__
from tja_ai_chartgen.audio.analyze import analyze_audio, apply_analysis_overrides
from tja_ai_chartgen.audio.convert import convert_to_ogg
from tja_ai_chartgen.features.bars import build_bar_features
from tja_ai_chartgen.features.sections import assign_sections
from tja_ai_chartgen.rules.fallback_generator import generate_fallback_chart_bars, validate_density
from tja_ai_chartgen.tja.model import ChartMetadata, SongAnalysis, TjaChart
from tja_ai_chartgen.tja.validator import ValidationIssue, validate_tja_text
from tja_ai_chartgen.tja.writer import render_tja
from tja_ai_chartgen.utils.paths import write_json

app = typer.Typer(help="AI-assisted TJA chart draft generator")
console = Console()


@app.command()
def version() -> None:
    console.print(f"tja-ai-chartgen {__version__}")


@app.command()
def generate(
    input_audio: Path,
    title: str = typer.Option(..., help="Song title written to the TJA metadata."),
    artist: str | None = typer.Option(None, help="Optional artist name."),
    output_dir: Path = typer.Option(Path("output"), help="Directory for generated files."),
    course: str = typer.Option("Oni", help="TJA course name."),
    level: int = typer.Option(10, help="TJA difficulty level."),
    style: str = typer.Option("technical", help="Draft generation style."),
    density: str = typer.Option(
        "auto",
        help="Draft density: auto, low, medium, high, or max.",
    ),
    max_bars: int | None = typer.Option(None, help="Only generate the first N bars."),
    bpm: float | None = typer.Option(None, help="Override analyzed BPM."),
    offset: float | None = typer.Option(None, help="Override analyzed OFFSET."),
    use_ai: bool = typer.Option(False, help="Use AI generation before falling back to rules."),
    model: str | None = typer.Option(None, help="Optional LiteLLM model name."),
) -> None:
    load_dotenv()
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_bars is not None and max_bars < 1:
        _fail("--max-bars must be greater than or equal to 1.")
    if bpm is not None and bpm <= 0:
        _fail("--bpm must be greater than 0.")
    try:
        validate_density(density)
    except ValueError as error:
        _fail(str(error))

    safe_stem = input_audio.stem
    ogg_path = output_dir / f"{safe_stem}.ogg"
    tja_path = output_dir / f"{safe_stem}.tja"
    analysis_path = output_dir / "analysis.json"
    generation_config_path = output_dir / "generation_config.json"
    ai_input_path = output_dir / "ai_input.json"
    ai_output_path = output_dir / "ai_output.json"
    report_path = output_dir / "report.txt"

    generation_config = _build_generation_config(
        input_audio=input_audio,
        title=title,
        artist=artist,
        output_dir=output_dir,
        course=course,
        level=level,
        style=style,
        density=density,
        max_bars=max_bars,
        bpm=bpm,
        offset=offset,
        use_ai=use_ai,
        model=model,
    )
    write_json(generation_config_path, generation_config)

    try:
        console.print(f"Converting audio: {input_audio} -> {ogg_path}")
        convert_to_ogg(input_audio, ogg_path)

        console.print("Analyzing audio...")
        raw = analyze_audio(ogg_path)
        raw = apply_analysis_overrides(raw, bpm=bpm, offset=offset)
        bars = assign_sections(build_bar_features(raw, max_bars=max_bars))
    except Exception as error:  # noqa: BLE001 - CLI boundary should hide tracebacks.
        _write_failure_report(report_path, str(error), generation_config_path)
        _fail(str(error))

    analysis = SongAnalysis(
        title=title,
        artist=artist,
        audio_file=str(input_audio),
        ogg_file=str(ogg_path),
        bpm=raw.bpm,
        offset=raw.offset,
        bars=bars,
    )
    write_json(analysis_path, analysis)

    chart_bars = None
    ai_failure: str | None = None

    if use_ai:
        try:
            from tja_ai_chartgen.ai.client import generate_chart_bars_with_ai, sanitize_ai_bars
            from tja_ai_chartgen.ai.prompts import build_chart_generation_payload

            write_json(ai_input_path, build_chart_generation_payload(analysis, course, level, style, density))
            ai_bars, ai_output = generate_chart_bars_with_ai(
                analysis,
                course,
                level,
                style,
                density,
                model,
            )
            write_json(ai_output_path, ai_output)
            chart_bars = sanitize_ai_bars(ai_bars, expected_count=len(bars))
        except Exception as error:  # noqa: BLE001 - CLI must keep producing a usable draft.
            ai_failure = str(error)
            console.print("AI generation failed. Falling back to rule-based generator.")
            write_json(ai_output_path, {"error": ai_failure})

    if chart_bars is None:
        chart_bars = generate_fallback_chart_bars(bars, style=style, density=density)

    chart = TjaChart(
        metadata=ChartMetadata(
            title=title,
            artist=artist,
            wave=ogg_path.name,
            bpm=analysis.bpm,
            offset=analysis.offset,
            course=course,
            level=level,
        ),
        bars=chart_bars,
    )
    tja_text = render_tja(chart)
    issues = validate_tja_text(tja_text)

    tja_path.write_text(tja_text, encoding="utf-8")
    _write_report(report_path, tja_path, analysis_path, generation_config_path, issues, ai_failure)

    _print_result(tja_path, analysis_path, generation_config_path, report_path, issues)

    if any(issue.level == "error" for issue in issues):
        raise typer.Exit(1)


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/] {message}")
    raise typer.Exit(1)


def _build_generation_config(
    *,
    input_audio: Path,
    title: str,
    artist: str | None,
    output_dir: Path,
    course: str,
    level: int,
    style: str,
    density: str,
    max_bars: int | None,
    bpm: float | None,
    offset: float | None,
    use_ai: bool,
    model: str | None,
) -> dict[str, Any]:
    return {
        "input_audio": str(input_audio),
        "title": title,
        "artist": artist,
        "output_dir": str(output_dir),
        "course": course,
        "level": level,
        "style": style,
        "density": density,
        "max_bars": max_bars,
        "bpm_override": bpm,
        "offset_override": offset,
        "use_ai": use_ai,
        "model": model,
    }


def _write_failure_report(
    report_path: Path,
    message: str,
    generation_config_path: Path | None = None,
) -> Path:
    lines = ["TJA AI Chart Generator Report", ""]
    if generation_config_path is not None:
        lines.extend([f"Generation config: {generation_config_path}", ""])
    lines.append(f"Error: {message}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _write_report(
    report_path: Path,
    tja_path: Path,
    analysis_path: Path,
    generation_config_path: Path,
    issues: list[ValidationIssue],
    ai_failure: str | None,
) -> Path:
    lines: list[str] = [
        "TJA AI Chart Generator Report",
        "",
        f"TJA: {tja_path}",
        f"Analysis: {analysis_path}",
        f"Generation config: {generation_config_path}",
        "",
    ]

    if ai_failure:
        lines.extend(["AI generation failed. Falling back to rule-based generator.", ai_failure, ""])

    if issues:
        lines.append("Validation issues:")
        for issue in issues:
            line = f"line {issue.line}: " if issue.line is not None else ""
            lines.append(f"[{issue.level}] {line}{issue.message}")
    else:
        lines.append("Validation passed.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _print_result(
    tja_path: Path,
    analysis_path: Path,
    generation_config_path: Path,
    report_path: Path,
    issues: list[ValidationIssue],
) -> None:
    console.print("Generated files:")
    console.print(f"- {tja_path}")
    console.print(f"- {analysis_path}")
    console.print(f"- {generation_config_path}")
    console.print(f"- {report_path}")

    for issue in issues:
        payload: dict[str, Any] = issue.model_dump()
        console.print(f"[{issue.level}] {payload}")


if __name__ == "__main__":
    app()
