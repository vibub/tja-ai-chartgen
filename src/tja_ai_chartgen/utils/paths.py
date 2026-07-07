import json
from pathlib import Path
from time import sleep
from typing import Any
from uuid import uuid4

from pydantic import BaseModel


def to_jsonable(data: Any) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump()
    if isinstance(data, list):
        return [to_jsonable(item) for item in data]
    if isinstance(data, dict):
        return {key: to_jsonable(value) for key, value in data.items()}
    return data


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(to_jsonable(data), ensure_ascii=False, indent=2)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp_path.write_text(payload, encoding="utf-8")
        for attempt in range(10):
            try:
                temp_path.replace(path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                sleep(0.01)
    finally:
        temp_path.unlink(missing_ok=True)
    return path
