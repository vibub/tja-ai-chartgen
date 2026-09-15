"""Inspect saved tracker timing and fit explicitly identified quarter-note anchors.

Tracker residuals are diagnostic evidence, not ground truth or automatic overrides.
Generated onset-grid candidates must never be used to validate their own clock.
"""

from math import isfinite
from statistics import median

from tja_ai_chartgen.tja.model import SongAnalysis

TIMING_DIAGNOSTIC_VERSION = "timing-diagnostic-v1"
TIMING_TOLERANCE_SECONDS = 0.03


def parse_timing_anchors(values: list[str]) -> list[tuple[float, float]]:
    anchors = []
    for value in values:
        try:
            beat, seconds = value.split(":")
            anchors.append((float(beat), float(seconds)))
        except ValueError as error:
            raise ValueError("Each --anchor must be BEAT:SECONDS, e.g. 0:0.25.") from error
    return anchors


def fit_timing_anchors(anchors: list[tuple[float, float]]) -> dict:
    """Beat 0 means the chart origin; one beat always means a quarter note."""
    if len(anchors) < 2:
        raise ValueError("Provide at least two anchors with different beat numbers.")
    points = sorted(anchors)
    if any(not isfinite(beat) or not isfinite(time) or time < 0 for beat, time in points):
        raise ValueError("Anchors must be finite; audio times must be nonnegative.")
    if any(b <= a or tb <= ta for (a, ta), (b, tb) in zip(points, points[1:])):
        raise ValueError("Anchor beat numbers and audio times must both increase strictly.")
    interval, offset = _linear_fit(points)
    if not isfinite(interval) or interval <= 0 or not isfinite(offset):
        raise ValueError("Anchor spacing cannot produce a finite positive tempo.")
    errors = [time - (offset + beat * interval) for beat, time in points]
    maximum = max(abs(error) for error in errors)
    return {
        "bpm": 60.0 / interval,
        "offset": offset,
        "tja_offset": -offset,
        "anchors": [{"beat": beat, "seconds": time} for beat, time in points],
        "max_fit_error_seconds": maximum,
        "consistent": maximum <= TIMING_TOLERANCE_SECONDS,
        "independently_checked": len(points) >= 3,
    }


def _linear_fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    variance = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance
    return slope, mean_y - slope * mean_x


def _residual_summary(
    points: list[tuple[float, float]], *, bpm: float, offset: float, duration: float,
) -> dict:
    residuals = [(time, time - (offset + beat * 60.0 / bpm)) for beat, time in points]
    windows = []
    for index, name in enumerate(("start", "middle", "end")):
        start, end = duration * index / 3, duration * (index + 1) / 3
        values = [
            residual for time, residual in residuals
            if start <= time and (time < end or (index == 2 and time <= end))
        ]
        windows.append({
            "region": name, "start_seconds": start, "end_seconds": end,
            "count": len(values),
            "median_error_seconds": median(values) if values else None,
        })
    errors = sorted(abs(residual) for _, residual in residuals)
    return {
        "count": len(points),
        "windows": windows,
        "median_error_seconds": median(r for _, r in residuals) if residuals else None,
        "p95_absolute_error_seconds": errors[min(len(errors) - 1, int(len(errors) * .95))]
        if errors else None,
        "points": [
            {"beat": beat, "seconds": time, "error_seconds": residual}
            for (beat, time), (_, residual) in zip(points, residuals)
        ],
    }


def build_timing_diagnostic(
    analysis: SongAnalysis, *, anchors: list[tuple[float, float]] | None = None,
) -> dict:
    if not isfinite(analysis.bpm) or analysis.bpm <= 0 or not isfinite(analysis.offset):
        raise ValueError("Analysis BPM must be positive and BPM/OFFSET must be finite.")
    duration = max((bar.end_time for bar in analysis.bars), default=0.0)
    calibration = fit_timing_anchors(anchors) if anchors else None
    if anchors:
        duration = max(duration, max(time for _, time in anchors))
    interval = 60.0 / analysis.bpm
    trackers = []
    for candidate in analysis.tempo_candidates:
        if candidate.source not in {"librosa", "beatnet"}:
            continue
        times = sorted({
            time for time in candidate.beat_times
            if isfinite(time) and 0 <= time <= duration
        })
        points = []
        for time in times:
            if not points:
                beat = round((time - analysis.offset) / interval)
            else:
                # Keep cumulative drift visible beyond half a beat, while allowing
                # missed beats and ignoring duplicate detections within one beat.
                increment = round((time - points[-1][1]) / interval)
                if increment <= 0:
                    continue
                beat = points[-1][0] + increment
            points.append((beat, time))
        summary = _residual_summary(
            points, bpm=analysis.bpm, offset=analysis.offset, duration=duration,
        )
        coverage = (points[-1][1] - points[0][1]) / duration if len(points) >= 2 else 0.0
        classification = "insufficient-evidence"
        start_error = summary["windows"][0]["median_error_seconds"]
        end_error = summary["windows"][2]["median_error_seconds"]
        if len(points) >= 8 and coverage >= .5 and start_error is not None and end_error is not None:
            if abs(candidate.bpm / analysis.bpm - 1) > .05:
                classification = "tempo-mismatch-or-alias"
            else:
                slope, intercept = _linear_fit(points)
                fit_errors = sorted(abs(time - (intercept + beat * slope)) for beat, time in points)
                if fit_errors[int(len(fit_errors) * .95)] > TIMING_TOLERANCE_SECONDS:
                    classification = "irregular-tracker"
                elif abs(end_error - start_error) > TIMING_TOLERANCE_SECONDS:
                    classification = "possible-bpm-drift"
                elif abs(summary["median_error_seconds"]) > TIMING_TOLERANCE_SECONDS:
                    classification = "possible-offset-shift"
                else:
                    classification = "aligned-to-tracker"
        trackers.append({
            "source": candidate.source,
            "tracker_bpm": candidate.bpm,
            "tracker_accepted": candidate.accepted,
            "classification": classification,
            "coverage": coverage,
            "ignored_duplicate_count": len(times) - len(points),
            **summary,
        })
    anchor_comparison = None
    if calibration is not None:
        anchor_comparison = {
            "before": _residual_summary(
                anchors, bpm=analysis.bpm, offset=analysis.offset, duration=duration,
            ),
            "after": _residual_summary(
                anchors, bpm=calibration["bpm"], offset=calibration["offset"], duration=duration,
            ),
        }
    return {
        "diagnostic_version": TIMING_DIAGNOSTIC_VERSION,
        "bpm": analysis.bpm, "offset": analysis.offset, "duration_seconds": duration,
        "trackers": trackers,
        "status": "observed-trackers" if trackers else "no-independent-tracker",
        "calibration": calibration,
        "anchor_comparison": anchor_comparison,
    }


def render_timing_markdown(report: dict) -> str:
    lines = [
        "# 时间轴诊断", "",
        f"当前 BPM：{report['bpm']:.6f}；内部 offset：{report['offset']:.6f} 秒。", "",
        f"诊断范围：0–{report['duration_seconds']:.3f} 秒，依据已保存的小节范围和人工锚点；"
        "截断生成的分析不代表全曲。", "",
        "正偏差表示音乐事件比当前网格晚，负偏差表示早。原始 tracker 不是人工真值；"
        "本报告不会自动修改时间轴，也不能单凭 beat 确认小节首拍或排除整拍歧义。", "",
        "tracker 的起始拍号按最近拍位推定，可能与明确标注拍号的人工锚点相差整数拍。", "",
        "| 来源 | 诊断 | 开头偏差 ms | 中段偏差 ms | 结尾偏差 ms |", "| --- | --- | ---: | ---: | ---: |",
    ]
    for tracker in report["trackers"]:
        values = [
            "—" if window["median_error_seconds"] is None
            else f"{window['median_error_seconds'] * 1000:+.1f}"
            for window in tracker["windows"]
        ]
        source = tracker["source"] + ("（候选已接受）" if tracker["tracker_accepted"] else "（候选未接受）")
        lines.append(f"| {source} | {tracker['classification']} | {' | '.join(values)} |")
    if not report["trackers"]:
        lines += ["", "没有独立 tracker；不使用等间隔合成节拍证明时间轴正确。"]
    lines += [
        "", "诊断说明：possible-offset-shift 为可能整体偏移；possible-bpm-drift 为可能持续漂移；"
        "irregular-tracker 为跟踪不规则；tempo-mismatch-or-alias 为速度或半速/倍速歧义；"
        "insufficient-evidence 为证据不足。以上均需结合试听核对。",
    ]
    calibration = report["calibration"]
    if calibration:
        lines += [
            "", "## 人工拍点校准", "",
            f"BPM：{calibration['bpm']:.9f}；内部 offset：{calibration['offset']:.9f} 秒；"
            f"TJA OFFSET：{calibration['tja_offset']:.9f} 秒。",
            f"锚点最大拟合误差：{calibration['max_fit_error_seconds'] * 1000:.1f} ms。",
            "拍号不影响计数单位：beat 0 是谱面起点，每 1 beat 始终代表一个四分音符。",
            "", "| 人工 beat | 音频秒数 | 校准前偏差 ms | 校准后偏差 ms |",
            "| ---: | ---: | ---: | ---: |",
        ]
        comparison = report["anchor_comparison"]
        for before, after in zip(comparison["before"]["points"], comparison["after"]["points"]):
            lines.append(
                f"| {before['beat']:g} | {before['seconds']:.6f} | "
                f"{before['error_seconds'] * 1000:+.1f} | {after['error_seconds'] * 1000:+.1f} |"
            )
        if not calibration["independently_checked"]:
            lines.append("只有两个锚点，零拟合误差不代表全曲准确；请增加一个中段锚点核验。")
        if not calibration["consistent"]:
            lines.append("锚点无法由固定 BPM 一致解释，请核对拍号计数或检查真实变速。")
    return "\n".join(lines) + "\n"
