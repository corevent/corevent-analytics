from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2),
        encoding="utf-8",
    )
