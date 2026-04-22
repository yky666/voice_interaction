from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from audio_io import AudioIO, AudioIOConfig


def _env_text(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


@dataclass(slots=True)
class LiveCompanionConfig:
    service_url: str
    session_id: str

    @classmethod
    def from_env(cls) -> "LiveCompanionConfig":
        return cls(
            service_url=_env_text("VOICE_SERVICE_TURN_URL", "http://127.0.0.1:8083/turn"),
            session_id=_env_text("VOICE_SESSION_ID", "robot-companion-session"),
        )


def _json_http_post(url: str, payload: dict[str, Any], timeout_sec: float = 120.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc


def main() -> int:
    cfg = LiveCompanionConfig.from_env()
    audio = AudioIO(AudioIOConfig.from_env())
    print(f"[voice] live companion started, session_id={cfg.session_id}, service={cfg.service_url}")
    print("[voice] press Ctrl+C to stop")
    while True:
        captured = audio.capture_until_silence()
        if not captured.get("speech_detected"):
            continue
        result = _json_http_post(
            cfg.service_url,
            {"session_id": cfg.session_id, "audio_b64": captured["audio_b64"]},
        )
        reply_text = str(result.get("reply_text") or "").strip()
        if reply_text:
            print(f"[assistant] {reply_text}")
        tts_result = result.get("tts_result") or {}
        audio_b64 = tts_result.get("audio_b64")
        if isinstance(audio_b64, str) and audio_b64.strip():
            audio.play_audio_b64_with_barge_in(audio_b64)
        media_path = str((result.get("llm_result") or {}).get("media_track_path") or "").strip()
        if media_path:
            print(f"[voice] playing remembered comfort track: {media_path}")
            audio.play_file_with_barge_in(media_path, delete_after=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[voice] stopped")
        raise SystemExit(0)
