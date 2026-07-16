import os
from ipaddress import ip_address
from pathlib import Path
from typing import Any, cast

import typer
from dotenv import load_dotenv
from rich.console import Console

from tja_ai_chartgen import __version__
from tja_ai_chartgen.audio.instrument_models import (
    InstrumentModelError,
    InstrumentModelProfile,
    prepare_instrument_models as prepare_local_instrument_models,
    resolve_instrument_model_dir,
)
from tja_ai_chartgen.features.meter import validate_time_signature
from tja_ai_chartgen.generation import (
    MAX_AI_TRANSPORT_RETRIES,
    GenerationConfig,
    GenerationNotice,
    build_analysis_notices,
    build_song_analysis,
    generate_chart_bars,
    load_generation_config,
)
from tja_ai_chartgen.rules.fallback_generator import validate_density
from tja_ai_chartgen.rules.styles import validate_style
from tja_ai_chartgen.tja.model import ChartMetadata, TjaChart
from tja_ai_chartgen.tja.validator import ValidationIssue, validate_tja_text
from tja_ai_chartgen.tja.writer import TjaEncodingError, render_tja, write_tja_text
from tja_ai_chartgen.utils.paths import write_json

app = typer.Typer(help="AI-assisted TJA chart draft generator")
console = Console()

COURSE_PRESETS = {
    "Easy": {"level": 3},
    "Normal": {"level": 5},
    "Hard": {"level": 7},
    "Oni": {"level": 10},
}
MULTI_COURSES = tuple(COURSE_PRESETS)
DEFAULT_AI_REQUEST_TIMEOUT = 300.0
DEFAULT_AI_TRANSPORT_RETRIES = 3


@app.command()
def version() -> None:
    console.print(f"tja-ai-chartgen {__version__}")


@app.command("prepare-instrument-models")
def prepare_instrument_models_command(
    model_dir: Path | None = typer.Option(
        None,
        "--model-dir",
        help="Local model directory. Defaults to the selected profile directory.",
    ),
    profile: str = typer.Option(
        "stem-role",
        "--profile",
        help=(
            "Model profile: stem-role (recommended Demucs rhythm enhancement) or "
            "full (legacy Demucs + AST taxonomy diagnostics)."
        ),
    ),
) -> None:
    if profile not in {"full", "stem-role"}:
        _fail("--profile must be full or stem-role.")
    resolved_profile = cast(InstrumentModelProfile, profile)
    target = resolve_instrument_model_dir(model_dir, profile=resolved_profile)
    capability = (
        "recommended stem-role rhythm enhancement"
        if resolved_profile == "stem-role"
        else "legacy full taxonomy diagnostics"
    )
    console.print(f"Preparing {capability} models in: {target}")
    try:
        manifest = prepare_local_instrument_models(target, profile=resolved_profile)
    except InstrumentModelError as error:
        _fail(f"Instrument model preparation failed: {error.reason}")
    except Exception as error:  # noqa: BLE001 - CLI boundary should hide tracebacks.
        _fail(f"Instrument model preparation failed: {type(error).__name__}")
    components = manifest.demucs_model
    if manifest.classifier_model is not None:
        components = f"{components}, {manifest.classifier_model}"
    console.print(f"Instrument models prepared ({manifest.profile}): {components}")
    if manifest.profile == "full":
        console.print(
            "Note: full is a legacy compatibility profile; AST taxonomy is diagnostic-only."
        )


@app.command()
def generate(
    input_audio: Path,
    title: str = typer.Option(..., help="Song title written to the TJA metadata."),
    artist: str | None = typer.Option(None, help="Optional artist name."),
    output_dir: Path = typer.Option(Path("output"), help="Directory for generated files."),
    course: str = typer.Option("Oni", help="TJA course name."),
    level: int = typer.Option(10, help="TJA difficulty level."),
    all_courses: bool = typer.Option(
        False,
        "--all-courses",
        help="Generate Easy, Normal, Hard, and Oni charts as separate TJA files.",
    ),
    style: str = typer.Option("technical", help="Draft generation style."),
    density: str = typer.Option(
        "auto",
        help="Draft density: auto, low, medium, high, or max.",
    ),
    max_bars: int | None = typer.Option(None, help="Only generate the first N bars."),
    bpm: float | None = typer.Option(None, help="Override analyzed BPM."),
    offset: float | None = typer.Option(None, help="Override analyzed OFFSET."),
    time_signature: str | None = typer.Option(
        None,
        "--time-signature",
        help="Override analyzed meter: 4/4, 3/4, or 6/8.",
    ),
    use_beatnet: bool = typer.Option(
        False,
        "--use-beatnet",
        help="Try optional BeatNet analysis for downbeat, meter, and bar start detection.",
    ),
    use_instrument_analysis: bool = typer.Option(
        False,
        "--use-instrument-analysis",
        help=(
            "Enable optional local stem-role rhythm enhancement. The legacy full "
            "profile additionally runs AST taxonomy diagnostics."
        ),
    ),
    instrument_profile: str = typer.Option(
        "stem-role",
        "--instrument-profile",
        help=(
            "Analysis profile: stem-role (recommended Demucs rhythm enhancement) or "
            "full (legacy Demucs + AST taxonomy diagnostics)."
        ),
    ),
    instrument_device: str = typer.Option(
        "auto",
        "--instrument-device",
        help="Instrument analysis device: auto, cpu, cuda, or mps.",
    ),
    instrument_model_dir: Path | None = typer.Option(
        None,
        "--instrument-model-dir",
        help="Local model directory. Defaults to the selected profile directory.",
    ),
    special_notes: bool = typer.Option(
        False,
        "--special-notes",
        help="Allow simple drumroll and balloon notes in generated drafts.",
    ),
    use_ai: bool = typer.Option(False, help="Use AI generation before falling back to rules."),
    model: str | None = typer.Option(None, help="Optional LiteLLM model name."),
    ai_base_url: str | None = typer.Option(
        None,
        "--ai-base-url",
        help="OpenAI-compatible API base URL passed to LiteLLM as api_base.",
    ),
    ai_api_key: str | None = typer.Option(
        None,
        "--ai-api-key",
        help="OpenAI-compatible API key. Prefer OPENAI_API_KEY in .env."
    ),
    ai_repair_retries: int = typer.Option(
        2,
        "--ai-repair-retries",
        help="Retry count for repairing invalid AI JSON output.",
    ),
    ai_request_timeout: float = typer.Option(
        DEFAULT_AI_REQUEST_TIMEOUT,
        "--ai-request-timeout",
        help="Timeout in seconds for each AI provider transport attempt (1-600).",
    ),
    ai_transport_retries: int = typer.Option(
        DEFAULT_AI_TRANSPORT_RETRIES,
        "--ai-transport-retries",
        help=f"Retry count shared by all AI provider call failures (0-{MAX_AI_TRANSPORT_RETRIES}).",
    ),
) -> None:
    run_generate(
        input_audio=input_audio,
        title=title,
        artist=artist,
        output_dir=output_dir,
        course=course,
        level=level,
        all_courses=all_courses,
        style=style,
        density=density,
        max_bars=max_bars,
        bpm=bpm,
        offset=offset,
        time_signature=time_signature,
        use_beatnet=use_beatnet,
        use_instrument_analysis=use_instrument_analysis,
        instrument_profile=instrument_profile,
        instrument_device=instrument_device,
        instrument_model_dir=instrument_model_dir,
        special_notes=special_notes,
        use_ai=use_ai,
        model=model,
        ai_base_url=ai_base_url,
        ai_api_key=ai_api_key,
        ai_repair_retries=ai_repair_retries,
        ai_request_timeout=ai_request_timeout,
        ai_transport_retries=ai_transport_retries,
    )


@app.command("generate-from-config")
def generate_from_config(config_path: Path) -> None:
    try:
        config = load_generation_config(config_path)
    except (FileNotFoundError, ValueError) as error:
        _fail(str(error) if isinstance(error, FileNotFoundError) else f"Invalid generation config: {error}")

    run_generate(
        input_audio=config.input_audio,
        title=config.title,
        artist=config.artist,
        output_dir=config.output_dir,
        course=config.course,
        level=config.level,
        all_courses=config.all_courses,
        style=config.style,
        density=config.density,
        max_bars=config.max_bars,
        bpm=config.bpm_override,
        offset=config.offset_override,
        time_signature=config.time_signature,
        use_beatnet=config.use_beatnet,
        use_instrument_analysis=config.use_instrument_analysis,
        instrument_profile=config.instrument_profile,
        instrument_device=config.instrument_device,
        instrument_model_dir=config.instrument_model_dir,
        special_notes=config.special_notes,
        use_ai=config.use_ai,
        model=config.model,
        ai_base_url=None,
        ai_api_key=None,
        ai_repair_retries=config.ai_repair_retries,
        ai_request_timeout=config.ai_request_timeout,
        ai_transport_retries=config.ai_transport_retries,
    )


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Host for the Web UI server."),
    port: int = typer.Option(8000, help="Port for the Web UI server."),
    output_dir: Path = typer.Option(Path("output/web"), help="Directory for Web UI jobs."),
    allow_remote: bool = typer.Option(
        False,
        "--allow-remote",
        help="Allow the unauthenticated Web UI to listen on a non-loopback address.",
    ),
    allow_instrument_analysis: bool = typer.Option(
        False,
        "--allow-instrument-analysis",
        help=(
            "Allow remote Web requests to run local stem-role enhancement and the "
            "optional legacy AST compatibility classifier."
        ),
    ),
) -> None:
    import asyncio
    import sys

    remote_mode = not _is_loopback_host(host)
    if remote_mode and not allow_remote:
        _fail("Non-loopback Web listening requires --allow-remote.")

    import uvicorn

    from tja_ai_chartgen.web import create_app

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(
        create_app(
            output_dir=output_dir,
            remote_mode=remote_mode,
            allow_instrument_analysis=allow_instrument_analysis,
        ),
        host=host,
        port=port,
        timeout_keep_alive=1,
        timeout_graceful_shutdown=1,
    )


def run_generate(
    *,
    input_audio: Path,
    title: str,
    artist: str | None,
    output_dir: Path,
    course: str,
    level: int,
    all_courses: bool,
    style: str,
    density: str,
    max_bars: int | None,
    bpm: float | None,
    offset: float | None,
    time_signature: str | None,
    use_beatnet: bool,
    use_instrument_analysis: bool,
    instrument_profile: str,
    instrument_device: str,
    instrument_model_dir: Path | None,
    special_notes: bool,
    use_ai: bool,
    model: str | None,
    ai_base_url: str | None,
    ai_api_key: str | None,
    ai_repair_retries: int,
    ai_request_timeout: float,
    ai_transport_retries: int,
) -> None:
    load_dotenv()
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_bars is not None and max_bars < 1:
        _fail("--max-bars must be greater than or equal to 1.")
    if bpm is not None and bpm <= 0:
        _fail("--bpm must be greater than 0.")
    if time_signature is not None:
        try:
            validate_time_signature(time_signature)
        except ValueError as error:
            _fail(str(error))
    if instrument_profile not in {"full", "stem-role"}:
        _fail("--instrument-profile must be full or stem-role.")
    resolved_instrument_profile = cast(InstrumentModelProfile, instrument_profile)
    if instrument_device not in {"auto", "cpu", "cuda", "mps"}:
        _fail("--instrument-device must be auto, cpu, cuda, or mps.")
    if ai_repair_retries < 0:
        _fail("--ai-repair-retries must be greater than or equal to 0.")
    if not 1 <= ai_request_timeout <= 600:
        _fail("--ai-request-timeout must be between 1 and 600 seconds.")
    if not 0 <= ai_transport_retries <= MAX_AI_TRANSPORT_RETRIES:
        _fail(
            "--ai-transport-retries must be between "
            f"0 and {MAX_AI_TRANSPORT_RETRIES}."
        )
    try:
        validate_density(density)
        validate_style(style)
    except ValueError as error:
        _fail(str(error))

    resolved_model = model or (os.getenv("MODEL", "openai/gpt-4o-mini") if use_ai else None)

    safe_stem = input_audio.stem
    ogg_path = output_dir / f"{safe_stem}.ogg"
    tja_path = output_dir / f"{safe_stem}.tja"
    analysis_path = output_dir / "analysis.json"
    generation_config_path = output_dir / "generation_config.json"
    ai_input_path = output_dir / "ai_input.json"
    ai_output_path = output_dir / "ai_output.json"
    quality_report_path = output_dir / "quality_report.json"
    notices_path = output_dir / "generation_notices.json"
    report_path = output_dir / "report.txt"

    generation_config = GenerationConfig(
        input_audio=input_audio,
        title=title,
        artist=artist,
        output_dir=output_dir,
        course=course,
        level=level,
        all_courses=all_courses,
        style=style,
        density=density,
        max_bars=max_bars,
        bpm_override=bpm,
        offset_override=offset,
        time_signature=time_signature,
        use_beatnet=use_beatnet,
        use_instrument_analysis=use_instrument_analysis,
        instrument_profile=resolved_instrument_profile,
        instrument_device=instrument_device,
        instrument_model_dir=instrument_model_dir,
        special_notes=special_notes,
        use_ai=use_ai,
        model=resolved_model,
        ai_repair_retries=ai_repair_retries,
        ai_request_timeout=ai_request_timeout,
        ai_transport_retries=ai_transport_retries,
    )
    write_json(generation_config_path, generation_config)

    try:
        console.print(f"Converting audio: {input_audio} -> {ogg_path}")
        analysis = build_song_analysis(
            input_audio=input_audio,
            ogg_path=ogg_path,
            title=title,
            artist=artist,
            max_bars=max_bars,
            bpm_override=bpm,
            offset_override=offset,
            time_signature_override=time_signature,
            use_beatnet=use_beatnet,
            use_instrument_analysis=use_instrument_analysis,
            instrument_profile=resolved_instrument_profile,
            instrument_device=instrument_device,
            instrument_model_dir=instrument_model_dir,
            stage_callback=lambda stage: console.print(
                (
                    "Running recommended Demucs stem-role rhythm enhancement..."
                    if resolved_instrument_profile == "stem-role"
                    else "Running legacy full Demucs + AST taxonomy diagnostics..."
                )
                if stage == "instruments"
                else "Analyzing audio rhythm and structure..."
            )
            if stage in {"analyze", "instruments"}
            else None,
        )
    except Exception as error:  # noqa: BLE001 - CLI boundary should hide tracebacks.
        _write_failure_report(report_path, str(error), generation_config_path)
        _fail(str(error))

    bars = analysis.bars
    write_json(analysis_path, analysis)

    course_specs = _build_course_specs(course, level, density, all_courses)
    tja_paths: list[Path] = []
    all_issues: list[ValidationIssue] = []
    ai_failures: list[str] = []
    notices = build_analysis_notices(
        analysis,
        requested_beatnet=use_beatnet,
        requested_instrument_analysis=use_instrument_analysis,
    )
    for notice in notices:
        _print_notice(notice)

    for course_spec in course_specs:
        course_name = str(course_spec["course"])
        course_level = int(course_spec["level"])
        course_density = str(course_spec["density"])
        course_tja_path = _course_tja_path(tja_path, course_name, all_courses)
        course_ai_input_path = _course_sidecar_path(ai_input_path, course_name, all_courses)
        course_ai_output_path = _course_sidecar_path(ai_output_path, course_name, all_courses)
        course_quality_report_path = _course_sidecar_path(
            quality_report_path, course_name, all_courses
        )
        course_ai_attempts_path = course_ai_output_path.with_name(
            course_ai_output_path.name.replace("ai_output", "ai_attempts", 1)
        )
        generation_result = generate_chart_bars(
            analysis=analysis,
            selected_bars=bars,
            course=course_name,
            level=course_level,
            style=style,
            density=course_density,
            special_notes=special_notes,
            use_ai=use_ai,
            model=resolved_model,
            api_base=ai_base_url,
            api_key=ai_api_key,
            ai_repair_retries=ai_repair_retries,
            ai_request_timeout=ai_request_timeout,
            ai_transport_retries=ai_transport_retries,
            ai_input_path=course_ai_input_path,
            ai_output_path=course_ai_output_path,
            ai_attempts_path=course_ai_attempts_path,
        )
        chart_bars = generation_result.chart_bars
        ai_failure = generation_result.ai_failure
        notices.extend(generation_result.notices)
        for notice in generation_result.notices:
            _print_notice(notice)
        if ai_failure:
            console.print(
                f"AI generation failed for {course_name}. Falling back to rule-based generator."
            )

        write_json(course_quality_report_path, generation_result.quality_report)
        chart = TjaChart(
            metadata=ChartMetadata(
                title=title,
                artist=artist,
                wave=ogg_path.name,
                bpm=analysis.bpm,
                offset=analysis.offset,
                course=course_name,
                level=course_level,
            ),
            bars=chart_bars,
        )
        tja_text = render_tja(chart)
        issues = validate_tja_text(tja_text)

        try:
            write_tja_text(course_tja_path, tja_text)
        except TjaEncodingError as error:
            _write_failure_report(report_path, str(error), generation_config_path)
            _fail(str(error))
        tja_paths.append(course_tja_path)
        all_issues.extend(issues)
        if ai_failure:
            ai_failures.append(f"{course_name}: {ai_failure}")

    write_json(notices_path, notices)
    _write_report(
        report_path,
        tja_paths,
        analysis_path,
        generation_config_path,
        all_issues,
        ai_failures,
        notices,
    )

    _print_result(
        tja_paths,
        analysis_path,
        generation_config_path,
        notices_path,
        report_path,
        all_issues,
    )

    if any(issue.level == "error" for issue in all_issues):
        raise typer.Exit(1)


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/] {message}")
    raise typer.Exit(1)


def _build_course_specs(
    course: str,
    level: int,
    density: str,
    all_courses: bool,
) -> list[dict[str, str | int]]:
    if not all_courses:
        return [{"course": course, "level": level, "density": density}]

    return [
        {"course": course_name, "level": int(preset["level"]), "density": density}
        for course_name, preset in COURSE_PRESETS.items()
    ]


def _course_tja_path(base_path: Path, course: str, all_courses: bool) -> Path:
    if not all_courses:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{course.lower()}{base_path.suffix}")


def _course_sidecar_path(base_path: Path, course: str, all_courses: bool) -> Path:
    if not all_courses:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{course.lower()}{base_path.suffix}")


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
    tja_paths: list[Path],
    analysis_path: Path,
    generation_config_path: Path,
    issues: list[ValidationIssue],
    ai_failures: list[str],
    notices: list[GenerationNotice],
) -> Path:
    lines: list[str] = [
        "TJA AI Chart Generator Report",
        "",
        "TJA:",
    ]
    lines.extend(f"- {tja_path}" for tja_path in tja_paths)
    lines.extend(
        [
            f"Analysis: {analysis_path}",
            f"Generation config: {generation_config_path}",
            "",
        ]
    )

    if ai_failures:
        lines.append("AI generation failed. Falling back to rule-based generator.")
        lines.extend(ai_failures)
        lines.append("")

    if notices:
        lines.append("Generation notices:")
        for notice in notices:
            scope = f" {notice.scope}" if notice.scope else ""
            lines.append(f"- [{notice.level}]{scope} {notice.message}")
            if notice.detail:
                lines.append(f"  {notice.detail}")
        lines.append("")

    if issues:
        lines.append("Validation issues:")
        for issue in issues:
            line = f"line {issue.line}: " if issue.line is not None else ""
            lines.append(f"[{issue.level}] {line}{issue.message}")
    else:
        lines.append("Validation passed.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _print_notice(notice: GenerationNotice) -> None:
    color = "red" if notice.level == "error" else "yellow" if notice.level == "warning" else "cyan"
    scope = f" {notice.scope}" if notice.scope else ""
    console.print(f"[{color}]{notice.level.upper()}{scope}:[/] {notice.message}")
    if notice.detail:
        console.print(f"  {notice.detail}")


def _print_result(
    tja_paths: list[Path],
    analysis_path: Path,
    generation_config_path: Path,
    notices_path: Path,
    report_path: Path,
    issues: list[ValidationIssue],
) -> None:
    console.print("Generated files:")
    for tja_path in tja_paths:
        console.print(f"- {tja_path}")
    console.print(f"- {analysis_path}")
    console.print(f"- {generation_config_path}")
    console.print(f"- {notices_path}")
    console.print(f"- {report_path}")

    for issue in issues:
        payload: dict[str, Any] = issue.model_dump()
        console.print(f"[{issue.level}] {payload}")


if __name__ == "__main__":
    app()
