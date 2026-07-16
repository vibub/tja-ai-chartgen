from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import socket
import sys
from threading import Event, Thread
from time import perf_counter
from typing import Any

from tja_ai_chartgen.audio.instrument_models import (
    InstrumentModelManifest,
    InstrumentModelProfile,
    validate_instrument_model_dir,
)
from tja_ai_chartgen.audio.instruments import InstrumentAnalysisRaw, analyze_instruments

INSTRUMENT_BENCHMARK_SCHEMA_VERSION = 1
INSTRUMENT_BENCHMARK_VERSION = "instrument-model-benchmark-v1"
_MEMORY_SAMPLE_INTERVAL_SECONDS = 0.01


def run_instrument_benchmark(
    audio_path: Path,
    *,
    model_dir: Path,
    profile: InstrumentModelProfile,
    device: str = "auto",
    analysis_sample_rate: int = 22_050,
    analysis_hop_length: int = 512,
    max_duration: float | None = 30.0,
    _analysis_runner: Callable[..., InstrumentAnalysisRaw] = analyze_instruments,
    _clock: Callable[[], float] = perf_counter,
    _rss_probe: Callable[[], int | None] | None = None,
) -> dict[str, Any]:
    """离线校验模型并测量一次真实 instrument analysis 的运行成本。"""
    if not audio_path.is_file():
        raise FileNotFoundError(f"Missing benchmark audio: {audio_path}")
    if max_duration is not None and max_duration <= 0:
        raise ValueError("max_duration must be positive")

    manifest = validate_instrument_model_dir(
        model_dir,
        profile=profile,
        verify_hashes=True,
    )
    inventory = build_model_inventory(model_dir, manifest)
    rss_probe = _rss_probe or current_process_rss_bytes
    sampler = _PeakRssSampler(rss_probe)
    rss_before = rss_probe()

    with _offline_runtime(), sampler:
        started_at = _clock()
        result = _analysis_runner(
            audio_path,
            model_dir=model_dir,
            profile=profile,
            device=device,
            analysis_sample_rate=analysis_sample_rate,
            analysis_hop_length=analysis_hop_length,
            max_duration=max_duration,
        )
        elapsed_seconds = max(0.0, _clock() - started_at)

    rss_after = rss_probe()
    peak_rss = sampler.peak_bytes
    expected_feature = "stem-role-v1" if profile == "stem-role" else "instrument-v1"
    passed = (
        result.status == "complete"
        and result.reason is None
        and result.feature_version == expected_feature
        and result.has_stem_evidence
        and (profile == "stem-role" or result.has_classifier_evidence)
    )
    analyzed_duration = float(result.analyzed_duration)
    realtime_factor = (
        elapsed_seconds / analyzed_duration if analyzed_duration > 0 else None
    )

    return {
        "schema_version": INSTRUMENT_BENCHMARK_SCHEMA_VERSION,
        "benchmark_version": INSTRUMENT_BENCHMARK_VERSION,
        "passed": passed,
        "profile": profile,
        "offline": {
            "network_blocked": True,
            "hf_hub_offline": True,
            "transformers_offline": True,
        },
        "model": inventory,
        "runtime": {
            "audio_file": audio_path.name,
            "requested_device": device,
            "resolved_device": result.device,
            "analysis_sample_rate": analysis_sample_rate,
            "analysis_hop_length": analysis_hop_length,
            "max_duration_seconds": max_duration,
            "analyzed_duration_seconds": round(analyzed_duration, 6),
            "elapsed_seconds": round(elapsed_seconds, 6),
            "realtime_factor": (
                round(realtime_factor, 6) if realtime_factor is not None else None
            ),
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "peak_rss_bytes": peak_rss,
            "peak_rss_delta_bytes": _memory_delta(peak_rss, rss_before),
        },
        "analysis": {
            "feature_version": result.feature_version,
            "status": result.status,
            "reason": result.reason,
            "demucs_model": result.demucs_model,
            "classifier_model": result.classifier_model,
            "stem_frame_count": len(result.stem_frames),
            "classification_window_count": len(result.classification_windows),
        },
    }


def build_model_inventory(
    model_dir: Path,
    manifest: InstrumentModelManifest,
) -> dict[str, Any]:
    component_bytes = {"demucs": 0, "ast": 0, "other": 0}
    for relative in manifest.files:
        candidate = model_dir / relative
        component = relative.split("/", 1)[0]
        if component not in component_bytes:
            component = "other"
        component_bytes[component] += candidate.stat().st_size
    return {
        "feature_version": manifest.feature_version,
        "manifest_profile": manifest.profile,
        "demucs_model": manifest.demucs_model,
        "demucs_revision": manifest.demucs_revision,
        "classifier_model": manifest.classifier_model,
        "classifier_revision": manifest.classifier_revision,
        "file_count": len(manifest.files),
        "total_bytes": sum(component_bytes.values()),
        "component_bytes": component_bytes,
        "hashes_verified": True,
        "code_licenses": manifest.code_licenses,
        "weight_licenses": manifest.weight_licenses,
    }


def render_instrument_benchmark_markdown(report: dict[str, Any]) -> str:
    runtime = report["runtime"]
    model = report["model"]
    analysis = report["analysis"]
    lines = [
        "# Instrument Model Benchmark",
        "",
        f"- 结果：{'PASS' if report['passed'] else 'FAIL'}",
        f"- Profile：`{report['profile']}`",
        f"- Feature：`{analysis['feature_version']}`",
        f"- 状态：`{analysis['status']}`",
        f"- 请求设备：`{runtime['requested_device']}`",
        f"- 实际设备：`{runtime['resolved_device']}`",
        "- 离线约束：socket blocked，HF Hub/Transformers offline",
        "",
        "## 模型成本",
        "",
        f"- 文件数：{model['file_count']}",
        f"- 总大小：{_format_bytes(model['total_bytes'])}",
        f"- Demucs：{_format_bytes(model['component_bytes']['demucs'])}",
        f"- AST：{_format_bytes(model['component_bytes']['ast'])}",
        "- Manifest SHA-256：已验证",
        "",
        "## 运行成本",
        "",
        f"- 分析音频：`{runtime['audio_file']}`",
        f"- 分析时长：{runtime['analyzed_duration_seconds']:.3f} s",
        f"- 墙钟耗时：{runtime['elapsed_seconds']:.3f} s",
        f"- Real-time factor：{_format_optional(runtime['realtime_factor'])}",
        f"- 峰值 RSS：{_format_bytes(runtime['peak_rss_bytes'])}",
        f"- 峰值 RSS 增量：{_format_bytes(runtime['peak_rss_delta_bytes'])}",
        "",
        "## 输出证据",
        "",
        f"- Stem frames：{analysis['stem_frame_count']}",
        f"- Classification windows：{analysis['classification_window_count']}",
    ]
    if analysis["reason"] is not None:
        lines.append(f"- Reason：`{analysis['reason']}`")
    lines.append("")
    return "\n".join(lines)


@contextmanager
def _offline_runtime() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    original_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")

    def blocked_connect(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("network access is disabled during instrument benchmark")

    def blocked_connect_ex(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("network access is disabled during instrument benchmark")

    socket.socket.connect = blocked_connect
    socket.socket.connect_ex = blocked_connect_ex
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        _restore_environment("HF_HUB_OFFLINE", original_hf_offline)
        _restore_environment("TRANSFORMERS_OFFLINE", original_transformers_offline)


class _PeakRssSampler:
    def __init__(self, probe: Callable[[], int | None]):
        self._probe = probe
        self._stop = Event()
        self._thread: Thread | None = None
        self.peak_bytes: int | None = None

    def __enter__(self) -> _PeakRssSampler:
        self.peak_bytes = self._probe()
        self._thread = Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._record(self._probe())

    def _sample(self) -> None:
        while not self._stop.wait(_MEMORY_SAMPLE_INTERVAL_SECONDS):
            self._record(self._probe())

    def _record(self, value: int | None) -> None:
        if value is not None and (self.peak_bytes is None or value > self.peak_bytes):
            self.peak_bytes = value


def current_process_rss_bytes() -> int | None:
    if sys.platform == "win32":
        return _windows_process_rss_bytes()
    proc_statm = Path("/proc/self/statm")
    if proc_statm.is_file():
        try:
            resident_pages = int(proc_statm.read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (IndexError, OSError, ValueError):
            return None
    try:
        import resource

        maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    return maximum_rss if sys.platform == "darwin" else maximum_rss * 1024


def _windows_process_rss_bytes() -> int | None:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
    except (AttributeError, OSError):
        return None
    return int(counters.WorkingSetSize) if success else None


def _restore_environment(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _memory_delta(peak: int | None, baseline: int | None) -> int | None:
    if peak is None or baseline is None:
        return None
    return max(0, peak - baseline)


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB")
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} GiB"


def _format_optional(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f}"
