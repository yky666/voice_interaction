from __future__ import annotations

import base64
import io
import json
import os
from typing import Any

import numpy as np
from PIL import Image


def getenv_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def getenv_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def safe_json_loads(text: str | None, default: Any = None) -> Any:
    if text is None or not text.strip():
        return default
    return json.loads(text)


def encode_image_b64(image: np.ndarray, format_name: str = "JPEG") -> str:
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    buffer = io.BytesIO()
    pil.save(buffer, format=format_name)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def serializable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serializable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return serializable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return serializable(vars(value))
        except Exception:
            pass
    return str(value)
