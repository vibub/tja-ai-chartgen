import json
from pathlib import Path
from typing import Any


AiSidecarPayload = dict[str, Any]


def load_ai_sidecar(path: Path) -> AiSidecarPayload:
    """按原始 JSON 对象读取历史或当前 AI sidecar，不改写旧字段。"""
    if not path.exists():
        raise FileNotFoundError(f"AI sidecar not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid AI sidecar JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("AI sidecar must be a JSON object.")
    return payload
