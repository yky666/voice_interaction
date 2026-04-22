from __future__ import annotations

import base64
import io
import importlib
import json
import math
import mimetypes
import os
import re
import shlex
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from common_utils import encode_image_b64, getenv_bool, getenv_int, safe_json_loads, serializable
from dialog_memory import ConversationMemoryStore


def _env_text(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def _json_http_post(
    url: str,
    payload: dict[str, Any],
    timeout_sec: float = 30.0,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Expected JSON response from {url}, got: {raw[:300]!r}") from exc


def _multipart_http_post(
    url: str,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
    timeout_sec: float = 60.0,
    headers: dict[str, str] | None = None,
) -> bytes:
    boundary = f"----voiceinteraction{uuid4().hex}"
    body = io.BytesIO()
    for key, value in fields.items():
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.write(str(value).encode("utf-8"))
        body.write(b"\r\n")
    for key, (filename, content, content_type) in files.items():
        guessed_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.write(f"Content-Type: {guessed_type}\r\n\r\n".encode("utf-8"))
        body.write(content)
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        url,
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc


def _run_json_command(command: str, payload: dict[str, Any], timeout_sec: float = 60.0) -> dict[str, Any]:
    if not command.strip():
        raise RuntimeError("Command backend is selected but no command is configured.")
    proc = subprocess.run(
        shlex.split(command),
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    raw = proc.stdout.strip()
    if not raw:
        raise RuntimeError("Command backend returned empty stdout; expected JSON.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Command backend returned invalid JSON: {raw[:300]!r}") from exc


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _coerce_text(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _parse_openai_chat_response(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenAI-compatible response did not contain choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        content = "\n".join(part for part in text_parts if part)
    text = str(content or "").strip()
    parsed = _extract_json_object(text)
    if parsed is not None:
        return parsed
    return {"reply_text": text}


def _guess_action_prompt(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    lowered = stripped.lower()
    action_markers = (
        "move",
        "pick",
        "grab",
        "open",
        "close",
        "raise",
        "lower",
        "turn",
        "wave",
        "reach",
        "抓",
        "拿",
        "抬",
        "移动",
        "伸手",
        "挥手",
        "转",
    )
    if any(marker in lowered for marker in action_markers):
        return stripped
    return ""


def _generate_mock_wav_base64(text: str, sample_rate: int = 16000) -> str:
    duration_sec = min(3.0, max(0.35, 0.045 * max(len(text), 1)))
    num_samples = max(1, int(sample_rate * duration_sec))
    frequency_hz = 440.0
    amplitude = 0.2
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for idx in range(num_samples):
            envelope = 1.0 - (idx / max(num_samples, 1)) * 0.15
            sample = amplitude * envelope * math.sin(2.0 * math.pi * frequency_hz * idx / sample_rate)
            wav_file.writeframes(struct.pack("<h", int(sample * 32767.0)))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass(slots=True)
class ASRConfig:
    mode: str
    command: str
    http_url: str
    timeout_sec: float
    mock_text: str
    api_base: str
    api_key: str
    model: str
    language: str
    device: str
    compute_type: str
    beam_size: int

    @classmethod
    def from_env(cls) -> "ASRConfig":
        return cls(
            mode=_env_text("VOICE_ASR_MODE", "mock").lower(),
            command=_env_text("VOICE_ASR_COMMAND", ""),
            http_url=_env_text("VOICE_ASR_HTTP_URL", ""),
            timeout_sec=float(_env_text("VOICE_ASR_TIMEOUT_SEC", "30")),
            mock_text=_env_text("VOICE_ASR_MOCK_TEXT", "小智，请介绍一下你自己。"),
            api_base=_env_text("VOICE_ASR_API_BASE", "https://api.openai.com/v1"),
            api_key=_env_text("VOICE_ASR_API_KEY", ""),
            model=_env_text("VOICE_ASR_MODEL", "gpt-4o-mini-transcribe"),
            language=_env_text("VOICE_ASR_LANGUAGE", "zh"),
            device=_env_text("VOICE_ASR_DEVICE", "auto"),
            compute_type=_env_text("VOICE_ASR_COMPUTE_TYPE", "auto"),
            beam_size=getenv_int("VOICE_ASR_BEAM_SIZE", 5),
        )


@dataclass(slots=True)
class LLMConfig:
    mode: str
    command: str
    http_url: str
    timeout_sec: float
    api_base: str
    api_key: str
    model: str
    temperature: float
    max_output_tokens: int

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            mode=_env_text("VOICE_LLM_MODE", "mock").lower(),
            command=_env_text("VOICE_LLM_COMMAND", ""),
            http_url=_env_text("VOICE_LLM_HTTP_URL", ""),
            timeout_sec=float(_env_text("VOICE_LLM_TIMEOUT_SEC", "60")),
            api_base=_env_text("VOICE_LLM_API_BASE", "https://api.openai.com/v1"),
            api_key=_env_text("VOICE_LLM_API_KEY", ""),
            model=_env_text("VOICE_LLM_MODEL", "gpt-4o-mini"),
            temperature=float(_env_text("VOICE_LLM_TEMPERATURE", "0.2")),
            max_output_tokens=getenv_int("VOICE_LLM_MAX_OUTPUT_TOKENS", 400),
        )


@dataclass(slots=True)
class TTSConfig:
    mode: str
    command: str
    http_url: str
    timeout_sec: float
    api_base: str
    api_key: str
    model: str
    voice: str
    audio_format: str

    @classmethod
    def from_env(cls) -> "TTSConfig":
        return cls(
            mode=_env_text("VOICE_TTS_MODE", "mock").lower(),
            command=_env_text("VOICE_TTS_COMMAND", ""),
            http_url=_env_text("VOICE_TTS_HTTP_URL", ""),
            timeout_sec=float(_env_text("VOICE_TTS_TIMEOUT_SEC", "60")),
            api_base=_env_text("VOICE_TTS_API_BASE", "https://api.openai.com/v1"),
            api_key=_env_text("VOICE_TTS_API_KEY", ""),
            model=_env_text("VOICE_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=_env_text("VOICE_TTS_VOICE", "alloy"),
            audio_format=_env_text("VOICE_TTS_AUDIO_FORMAT", "wav"),
        )


@dataclass(slots=True)
class DialogManagerConfig:
    system_prompt: str
    style_prompt: str
    response_format_prompt: str
    recent_history_limit: int
    enable_action_forward: bool
    action_service_url: str
    action_timeout_sec: float
    enable_g1_execute: bool
    g1_execute_url: str
    g1_timeout_sec: float
    default_image_b64: str
    wake_word_enabled: bool
    wake_word: str
    wake_duration_sec: int
    enable_comfort_actions: bool
    enable_song_memory: bool
    media_library: dict[str, str]
    enable_skill_router: bool
    g1_program_url: str
    skill_program_map: dict[str, str]
    skill_execute_motion: bool
    skill_dt_sec: float
    skill_scale: float
    skill_repeat: int

    @classmethod
    def from_env(cls) -> "DialogManagerConfig":
        default_format = (
            "Return JSON only with keys: "
            "reply_text, action_prompt, should_forward_action, should_execute_motion, "
            "skill_name, skill_args, memory_key, memory_value."
        )
        return cls(
            system_prompt=_env_text(
                "VOICE_SYSTEM_PROMPT",
                "You are a polite, safety-aware robot assistant. Speak briefly, clearly, and helpfully.",
            ),
            style_prompt=_env_text(
                "VOICE_STYLE_PROMPT",
                "Use a calm, friendly, and concise speaking style. If an action is unsafe or unclear, ask for confirmation.",
            ),
            response_format_prompt=_env_text("VOICE_RESPONSE_FORMAT_PROMPT", default_format),
            recent_history_limit=getenv_int("VOICE_MEMORY_RECENT_LIMIT", 12),
            enable_action_forward=getenv_bool("VOICE_ENABLE_ACTION_FORWARD", True),
            action_service_url=_env_text("VOICE_ACTION_SERVICE_URL", "http://127.0.0.1:8081/infer"),
            action_timeout_sec=float(_env_text("VOICE_ACTION_TIMEOUT_SEC", "60")),
            enable_g1_execute=getenv_bool("VOICE_ENABLE_G1_EXECUTE", False),
            g1_execute_url=_env_text("VOICE_G1_EXECUTE_URL", "http://127.0.0.1:8082/execute"),
            g1_timeout_sec=float(_env_text("VOICE_G1_TIMEOUT_SEC", "60")),
            default_image_b64=_env_text("VOICE_DEFAULT_IMAGE_B64", ""),
            wake_word_enabled=getenv_bool("VOICE_WAKE_WORD_ENABLED", False),
            wake_word=_env_text("VOICE_WAKE_WORD", "小智"),
            wake_duration_sec=getenv_int("VOICE_WAKE_DURATION_SEC", 30),
            enable_comfort_actions=getenv_bool("VOICE_ENABLE_COMFORT_ACTIONS", True),
            enable_song_memory=getenv_bool("VOICE_ENABLE_SONG_MEMORY", True),
            media_library=safe_json_loads(os.getenv("VOICE_MEDIA_LIBRARY_JSON"), default={}) or {},
            enable_skill_router=getenv_bool("VOICE_ENABLE_SKILL_ROUTER", True),
            g1_program_url=_env_text("VOICE_G1_PROGRAM_URL", "http://127.0.0.1:8082/run_program"),
            skill_program_map=safe_json_loads(
                os.getenv("VOICE_SKILL_PROGRAM_MAP"),
                default={
                    "comfort_hug": "sdk_arm_hug",
                    "handshake": "sdk_arm_shake_hand",
                    "high_five": "sdk_arm_high_five",
                    "greet_bow": "greet_bow",
                    "left_wave": "left_wave",
                    "right_wave": "right_wave",
                },
            )
            or {},
            skill_execute_motion=getenv_bool("VOICE_SKILL_EXECUTE_MOTION", False),
            skill_dt_sec=float(_env_text("VOICE_SKILL_DT_SEC", "0.05")),
            skill_scale=float(_env_text("VOICE_SKILL_SCALE", "1.0")),
            skill_repeat=getenv_int("VOICE_SKILL_REPEAT", 1),
        )


class ASRBackend:
    def __init__(self, config: ASRConfig) -> None:
        self.config = config
        self._fw_model: Any | None = None

    def transcribe(self, audio_b64: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        if self.config.mode == "mock":
            text = _coerce_text(payload, "asr_hint", "mock_text", "user_text") or self.config.mock_text
            return {"backend": "mock", "text": text}
        if self.config.mode == "faster_whisper_local":
            return self._transcribe_faster_whisper(audio_b64)
        if self.config.mode == "command":
            return {"backend": "command", **_run_json_command(self.config.command, {"audio_b64": audio_b64, **payload}, self.config.timeout_sec)}
        if self.config.mode == "http":
            return {"backend": "http", **_json_http_post(self.config.http_url, {"audio_b64": audio_b64, **payload}, self.config.timeout_sec)}
        if self.config.mode == "openai_compatible":
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            response_bytes = _multipart_http_post(
                f"{self.config.api_base.rstrip('/')}/audio/transcriptions",
                fields={
                    "model": self.config.model,
                    "language": self.config.language,
                    "response_format": "json",
                },
                files={"file": ("audio.wav", base64.b64decode(audio_b64), "audio/wav")},
                timeout_sec=self.config.timeout_sec,
                headers=headers,
            )
            data = json.loads(response_bytes.decode("utf-8"))
            return {"backend": "openai_compatible", **data}
        raise RuntimeError(f"Unsupported ASR mode: {self.config.mode}")

    def _transcribe_faster_whisper(self, audio_b64: str) -> dict[str, Any]:
        model = self._get_faster_whisper_model()
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            segments, info = model.transcribe(
                tmp_path,
                language=self.config.language or None,
                beam_size=max(1, int(self.config.beam_size)),
                vad_filter=True,
            )
            text_parts = [str(segment.text).strip() for segment in segments if str(segment.text).strip()]
            return {
                "backend": "faster_whisper_local",
                "text": " ".join(text_parts).strip(),
                "language": getattr(info, "language", self.config.language),
                "duration": float(getattr(info, "duration", 0.0) or 0.0),
                "duration_after_vad": float(getattr(info, "duration_after_vad", 0.0) or 0.0),
            }
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _get_faster_whisper_model(self) -> Any:
        if self._fw_model is not None:
            return self._fw_model
        faster_whisper = importlib.import_module("faster_whisper")
        device = None if self.config.device == "auto" else self.config.device
        compute_type = None if self.config.compute_type == "auto" else self.config.compute_type
        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        if compute_type:
            kwargs["compute_type"] = compute_type
        self._fw_model = faster_whisper.WhisperModel(self.config.model, **kwargs)
        return self._fw_model


class LLMBackend:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def generate(self, messages: list[dict[str, str]], context: dict[str, Any]) -> dict[str, Any]:
        if self.config.mode == "mock":
            return self._mock_generate(context)
        if self.config.mode == "command":
            return {"backend": "command", **_run_json_command(self.config.command, {"messages": messages, "context": context}, self.config.timeout_sec)}
        if self.config.mode == "http":
            return {"backend": "http", **_json_http_post(self.config.http_url, {"messages": messages, "context": context}, self.config.timeout_sec)}
        if self.config.mode == "openai_compatible":
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            data = _json_http_post(
                f"{self.config.api_base.rstrip('/')}/chat/completions",
                {
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_output_tokens,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                },
                timeout_sec=self.config.timeout_sec,
                headers=headers,
            )
            return {"backend": "openai_compatible", **_parse_openai_chat_response(data)}
        raise RuntimeError(f"Unsupported LLM mode: {self.config.mode}")

    def _mock_generate(self, context: dict[str, Any]) -> dict[str, Any]:
        user_text = str(context.get("user_text") or "").strip()
        action_prompt = _guess_action_prompt(user_text)
        facts = context.get("facts") or {}
        preferred_name = str(facts.get("preferred_name", "")).strip() if isinstance(facts, dict) else ""
        opening = f"{preferred_name}，" if preferred_name else ""
        reply_text = f"{opening}我已收到你的请求：{user_text}" if user_text else f"{opening}我已经准备好了，请告诉我接下来要做什么。"
        if action_prompt:
            reply_text += " 我会先整理成机器人动作指令并转发给执行链路。"
        memory_key = ""
        memory_value = ""
        name_match = re.search(r"(我叫|我的名字是)([^，。,.!！?？]{1,20})", user_text)
        if name_match:
            memory_key = "preferred_name"
            memory_value = name_match.group(2).strip()
            reply_text = f"记住了，之后我会称呼你为{memory_value}。"
        return {
            "backend": "mock",
            "reply_text": reply_text,
            "action_prompt": action_prompt,
            "should_forward_action": bool(action_prompt),
            "should_execute_motion": False,
            "skill_name": "robot_action" if action_prompt else "",
            "skill_args": {"prompt": action_prompt} if action_prompt else {},
            "memory_key": memory_key,
            "memory_value": memory_value,
        }


class TTSBackend:
    def __init__(self, config: TTSConfig) -> None:
        self.config = config

    def synthesize(self, text: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        if self.config.mode == "mock":
            return {"backend": "mock", "audio_b64": _generate_mock_wav_base64(text), "audio_format": "wav", "text": text}
        if self.config.mode == "command":
            return {"backend": "command", **_run_json_command(self.config.command, {"text": text, **payload}, self.config.timeout_sec)}
        if self.config.mode == "http":
            return {"backend": "http", **_json_http_post(self.config.http_url, {"text": text, **payload}, self.config.timeout_sec)}
        if self.config.mode == "openai_compatible":
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            request = urllib.request.Request(
                f"{self.config.api_base.rstrip('/')}/audio/speech",
                data=json.dumps(
                    {
                        "model": self.config.model,
                        "voice": self.config.voice,
                        "format": self.config.audio_format,
                        "input": text,
                    }
                ).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                    audio_bytes = response.read()
            except urllib.error.HTTPError as exc:
                body_text = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code} from {request.full_url}: {body_text}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Failed to reach {request.full_url}: {exc.reason}") from exc
            return {
                "backend": "openai_compatible",
                "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                "audio_format": self.config.audio_format,
                "text": text,
            }
        raise RuntimeError(f"Unsupported TTS mode: {self.config.mode}")


class DialogOrchestrator:
    def __init__(
        self,
        asr_backend: ASRBackend,
        llm_backend: LLMBackend,
        tts_backend: TTSBackend,
        memory_store: ConversationMemoryStore,
        config: DialogManagerConfig,
    ) -> None:
        self.asr_backend = asr_backend
        self.llm_backend = llm_backend
        self.tts_backend = tts_backend
        self.memory_store = memory_store
        self.config = config

    def health(self) -> dict[str, Any]:
        return {
            "asr_mode": self.asr_backend.config.mode,
            "llm_mode": self.llm_backend.config.mode,
            "tts_mode": self.tts_backend.config.mode,
            "action_forward_enabled": self.config.enable_action_forward,
            "action_service_url": self.config.action_service_url,
            "g1_execute_enabled": self.config.enable_g1_execute,
            "g1_execute_url": self.config.g1_execute_url,
            "wake_word_enabled": self.config.wake_word_enabled,
            "wake_word": self.config.wake_word,
            "wake_duration_sec": self.config.wake_duration_sec,
            "comfort_actions_enabled": self.config.enable_comfort_actions,
            "song_memory_enabled": self.config.enable_song_memory,
            "skill_router_enabled": self.config.enable_skill_router,
            "g1_program_url": self.config.g1_program_url,
            "skill_program_map": self.config.skill_program_map,
        }

    def create_session_id(self) -> str:
        return uuid4().hex

    def handle_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id") or self.create_session_id()).strip()
        user_text = str(payload.get("user_text") or payload.get("text") or "").strip()
        audio_b64 = str(payload.get("audio_b64") or "").strip()
        image_b64 = str(payload.get("image_b64") or self.config.default_image_b64).strip()
        wrist_image_b64 = str(payload.get("wrist_image_b64") or "").strip()
        robot_joints = payload.get("robot_joints")
        robot_state = payload.get("robot_state")
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

        asr_result: dict[str, Any] | None = None
        if not user_text and audio_b64:
            asr_result = self.asr_backend.transcribe(audio_b64, payload=metadata)
            user_text = _coerce_text(asr_result, "text", "transcript")
        if not user_text:
            raise ValueError("Missing user_text/text, and ASR did not produce a transcript.")

        wake_state = self._check_wake_word(session_id=session_id, user_text=user_text)
        if not wake_state["active"]:
            return {
                "ok": True,
                "session_id": session_id,
                "user_text": user_text,
                "asr_result": serializable(asr_result),
                "wake_word": wake_state,
                "reply_text": "",
                "tts_result": None,
                "action_result": None,
                "g1_result": None,
                "memory": self.memory_store.summarize_context(session_id, recent_limit=self.config.recent_history_limit),
            }

        history = self.memory_store.get_recent_messages(session_id, limit=self.config.recent_history_limit)
        facts = self.memory_store.get_facts(session_id)
        learned_facts = self._extract_companion_facts(user_text) if self.config.enable_song_memory else {}
        for key, value in learned_facts.items():
            self.memory_store.set_fact(session_id, key, value)
        if learned_facts:
            facts.update(learned_facts)
        llm_messages = self._build_messages(history=history, facts=facts, user_text=user_text)
        llm_result = self.llm_backend.generate(
            llm_messages,
            {"session_id": session_id, "user_text": user_text, "facts": facts, "metadata": metadata},
        )
        llm_result = self._apply_companion_postprocess(user_text=user_text, facts=facts, llm_result=llm_result)

        reply_text = _coerce_text(llm_result, "reply_text", "text", "response")
        if not reply_text:
            reply_text = "我已经收到你的请求，但当前没有生成可播报的回复。"

        memory_key = _coerce_text(llm_result, "memory_key")
        memory_value = _coerce_text(llm_result, "memory_value")
        if memory_key and memory_value:
            self.memory_store.set_fact(session_id, memory_key, memory_value)

        self.memory_store.append_message(session_id, "user", user_text, {"source": "audio" if audio_b64 else "text", "metadata": metadata})
        self.memory_store.append_message(session_id, "assistant", reply_text, {"llm": llm_result.get("backend", self.llm_backend.config.mode)})

        tts_result = self.tts_backend.synthesize(reply_text, payload=metadata)

        action_result: dict[str, Any] | None = None
        g1_result: dict[str, Any] | None = None
        skill_result: dict[str, Any] | None = None
        action_prompt = _coerce_text(llm_result, "action_prompt")
        should_forward_action = bool(llm_result.get("should_forward_action", bool(action_prompt)))
        should_execute_motion = bool(llm_result.get("should_execute_motion", False))
        skill_name = _coerce_text(llm_result, "skill_name")
        skill_args = llm_result.get("skill_args") if isinstance(llm_result.get("skill_args"), dict) else {}

        skill_consumed = False
        if self.config.enable_skill_router and skill_name:
            skill_result = self._call_skill_router(skill_name=skill_name, skill_args=skill_args)
            skill_consumed = bool(skill_result.get("ok")) or "error" in skill_result

        if self.config.enable_action_forward and should_forward_action and action_prompt and not skill_consumed:
            action_result = self._call_action_service(
                prompt=action_prompt,
                image_b64=image_b64,
                wrist_image_b64=wrist_image_b64,
                robot_joints=robot_joints,
                robot_state=robot_state,
                metadata={"session_id": session_id, "skill_name": skill_name, **metadata},
            )
            if self.config.enable_g1_execute and should_execute_motion and action_result:
                g1_result = self._call_g1_service(action_result)

        return {
            "ok": True,
            "session_id": session_id,
            "user_text": user_text,
            "asr_result": serializable(asr_result),
            "wake_word": wake_state,
            "llm_result": serializable(llm_result),
            "reply_text": reply_text,
            "tts_result": serializable(tts_result),
            "skill_result": serializable(skill_result),
            "action_result": serializable(action_result),
            "g1_result": serializable(g1_result),
            "memory": self.memory_store.summarize_context(session_id, recent_limit=self.config.recent_history_limit),
        }

    def _check_wake_word(self, session_id: str, user_text: str) -> dict[str, Any]:
        if not self.config.wake_word_enabled:
            return {"enabled": False, "active": True, "matched": False, "wake_word": self.config.wake_word}
        now = time.time()
        facts = self.memory_store.get_facts(session_id)
        wake_until_raw = facts.get("wake_until", "")
        try:
            wake_until = float(wake_until_raw) if wake_until_raw else 0.0
        except Exception:
            wake_until = 0.0
        matched = self.config.wake_word in user_text
        if matched:
            wake_until = now + float(self.config.wake_duration_sec)
            self.memory_store.set_fact(session_id, "wake_until", str(wake_until))
        active = matched or wake_until > now
        return {
            "enabled": True,
            "active": active,
            "matched": matched,
            "wake_word": self.config.wake_word,
            "wake_until": wake_until,
        }

    def _build_messages(self, history: list[dict[str, Any]], facts: dict[str, str], user_text: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        fact_lines = [f"- {key}: {value}" for key, value in sorted(facts.items()) if key != "wake_until"]
        system_parts = [self.config.system_prompt, self.config.style_prompt, self.config.response_format_prompt]
        if fact_lines:
            system_parts.append("Known user facts:\n" + "\n".join(fact_lines))
        messages.append({"role": "system", "content": "\n\n".join(part for part in system_parts if part)})
        for item in history:
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if role in {"user", "assistant", "system"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})
        return messages

    def _extract_companion_facts(self, user_text: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        patterns = [
            (r"(我伤心的时候想听|我难过的时候想听)(《?[^，。,.!！?？]{1,30}》?)", "comfort_song_sad"),
            (r"(我焦虑的时候想听|我紧张的时候想听)(《?[^，。,.!！?？]{1,30}》?)", "comfort_song_anxious"),
            (r"(我开心的时候想听)(《?[^，。,.!！?？]{1,30}》?)", "comfort_song_happy"),
        ]
        for pattern, key in patterns:
            match = re.search(pattern, user_text)
            if match:
                facts[key] = match.group(2).strip(" ，。")
        return facts

    def _apply_companion_postprocess(
        self,
        user_text: str,
        facts: dict[str, str],
        llm_result: dict[str, Any],
    ) -> dict[str, Any]:
        emotion = _detect_emotion(user_text)
        reply_text = _coerce_text(llm_result, "reply_text", "text", "response")
        action_prompt = _coerce_text(llm_result, "action_prompt")

        if emotion == "sad":
            song_name = str(facts.get("comfort_song_sad", "")).strip()
            if song_name and song_name not in reply_text:
                prefix = f"{reply_text} " if reply_text else ""
                reply_text = prefix + f"我记得你难过的时候想听{song_name}，如果你愿意，我可以先陪着你，也可以准备播放这首歌。"
            elif not reply_text:
                reply_text = "我在这儿。你可以慢慢说，我会陪着你。"
            if self.config.enable_comfort_actions and not _coerce_text(llm_result, "skill_name"):
                action_prompt = "Offer a brief comforting hug posture."
                llm_result["should_forward_action"] = False
                llm_result["should_execute_motion"] = self.config.skill_execute_motion
                llm_result["skill_name"] = "comfort_hug"
                llm_result["skill_args"] = {"emotion": "sad"}
            media_path = self._resolve_media_track(song_name)
            if media_path:
                llm_result["media_track_name"] = song_name
                llm_result["media_track_path"] = media_path

        if emotion == "anxious" and not reply_text:
            reply_text = "先别着急，我们慢慢来。你说一句，我陪你理一句。"
        if emotion == "happy" and not reply_text:
            reply_text = "听起来你现在状态不错，我也很为你开心。"

        if reply_text:
            llm_result["reply_text"] = reply_text
        if action_prompt:
            llm_result["action_prompt"] = action_prompt
        llm_result["emotion"] = emotion
        return llm_result

    def _resolve_media_track(self, song_name: str) -> str:
        if not song_name:
            return ""
        raw = self.config.media_library.get(song_name) or self.config.media_library.get(song_name.strip("《》"))
        if not raw:
            return ""
        path = str(Path(raw).expanduser())
        return path if Path(path).exists() else ""

    def _call_action_service(
        self,
        prompt: str,
        image_b64: str,
        wrist_image_b64: str,
        robot_joints: Any,
        robot_state: Any,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        final_image_b64 = image_b64 or encode_image_b64(_black_image())
        payload = {
            "image_b64": final_image_b64,
            "wrist_image_b64": wrist_image_b64 or None,
            "robot_joints": robot_joints,
            "robot_state": robot_state,
            "prompt": prompt,
            "metadata": metadata,
        }
        return _json_http_post(self.config.action_service_url, payload, timeout_sec=self.config.action_timeout_sec)

    def _call_g1_service(self, action_result: dict[str, Any]) -> dict[str, Any]:
        return _json_http_post(
            self.config.g1_execute_url,
            {"execute_motion": True, "pi05_result": action_result},
            timeout_sec=self.config.g1_timeout_sec,
        )

    def _call_skill_router(self, skill_name: str, skill_args: dict[str, Any]) -> dict[str, Any]:
        program_id = self.config.skill_program_map.get(skill_name)
        if not program_id:
            return {"ok": False, "skill_name": skill_name, "error": f"No mapped program for skill_name={skill_name!r}"}
        payload = {
            "program_id": program_id,
            "execute_motion": bool(skill_args.get("execute_motion", self.config.skill_execute_motion)),
            "dt_sec": float(skill_args.get("dt_sec", self.config.skill_dt_sec)),
            "scale": float(skill_args.get("scale", self.config.skill_scale)),
            "repeat": int(skill_args.get("repeat", self.config.skill_repeat)),
        }
        result = _json_http_post(self.config.g1_program_url, payload, timeout_sec=self.config.g1_timeout_sec)
        return {
            "ok": bool(result.get("ok", False)),
            "skill_name": skill_name,
            "program_id": program_id,
            "request": payload,
            "result": result,
        }


def _black_image() -> Any:
    import numpy as np

    return np.zeros((224, 224, 3), dtype=np.uint8)


def _detect_emotion(text: str) -> str:
    lowered = text.strip().lower()
    if not lowered:
        return "neutral"
    sad_markers = ("难过", "伤心", "想哭", "失落", "心情不好", "孤独", "崩溃", "sad", "upset")
    anxious_markers = ("焦虑", "紧张", "害怕", "慌", "压力大", "anxious", "panic")
    happy_markers = ("开心", "高兴", "兴奋", "幸福", "快乐", "happy")
    angry_markers = ("生气", "愤怒", "烦死了", "讨厌", "angry", "mad")
    if any(marker in lowered for marker in sad_markers):
        return "sad"
    if any(marker in lowered for marker in anxious_markers):
        return "anxious"
    if any(marker in lowered for marker in happy_markers):
        return "happy"
    if any(marker in lowered for marker in angry_markers):
        return "angry"
    return "neutral"
