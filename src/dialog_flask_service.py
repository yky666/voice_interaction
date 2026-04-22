from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from common_utils import getenv_bool
from dialog_memory import ConversationMemoryStore
from dialog_pipeline import ASRBackend, ASRConfig, DialogManagerConfig, DialogOrchestrator, LLMBackend, LLMConfig, TTSBackend, TTSConfig

app = Flask(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
MEMORY_DB_PATH = os.getenv("VOICE_MEMORY_DB_PATH", "").strip() or str(ROOT_DIR / "voice_memory.sqlite3")

ORCHESTRATOR = DialogOrchestrator(
    asr_backend=ASRBackend(ASRConfig.from_env()),
    llm_backend=LLMBackend(LLMConfig.from_env()),
    tts_backend=TTSBackend(TTSConfig.from_env()),
    memory_store=ConversationMemoryStore(MEMORY_DB_PATH),
    config=DialogManagerConfig.from_env(),
)


@app.after_request
def add_cors_headers(response: Any) -> Any:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/<path:_unused>", methods=["OPTIONS"])
def options_any(_unused: str) -> Any:
    return ("", 204)


@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True, "service": "voice_interaction_service", "memory_db_path": MEMORY_DB_PATH, "pipeline": ORCHESTRATOR.health()})


@app.get("/example_schema")
def example_schema() -> Any:
    return jsonify(
        {
            "request": {
                "session_id": "optional-stable-session-id",
                "user_text": "小智，请挥挥手，然后介绍一下你自己",
                "audio_b64": "<optional, mono wav or other accepted format>",
                "image_b64": "<optional, current head camera frame>",
                "wrist_image_b64": "<optional, wrist camera frame>",
                "robot_joints": [0.0, 0.1, -0.2],
                "robot_state": {"joint_positions": [0.0, 0.1, -0.2]},
                "metadata": {"robot": "unitree_g1"},
            },
            "response": {
                "ok": True,
                "session_id": "generated-or-input-session-id",
                "reply_text": "我已收到你的请求……",
                "tts_result": {"audio_b64": "<base64 wav>", "audio_format": "wav"},
                "skill_result": {"ok": True, "program_id": "sdk_arm_hug"},
                "action_result": {"ok": True},
            },
        }
    )


@app.post("/turn")
def turn() -> Any:
    payload = request.get_json(force=True, silent=False) or {}
    return jsonify(ORCHESTRATOR.handle_turn(payload))


@app.post("/memory/fact")
def set_memory_fact() -> Any:
    payload = request.get_json(force=True, silent=False) or {}
    session_id = str(payload.get("session_id", "")).strip()
    key = str(payload.get("key", "")).strip()
    value = str(payload.get("value", "")).strip()
    if not session_id:
        raise ValueError("Missing required field: session_id")
    if not key:
        raise ValueError("Missing required field: key")
    if not value:
        raise ValueError("Missing required field: value")
    ORCHESTRATOR.memory_store.set_fact(session_id, key, value)
    return jsonify({"ok": True, "service": "voice_interaction_service", "session_id": session_id, "facts": ORCHESTRATOR.memory_store.get_facts(session_id)})


@app.get("/memory/<session_id>")
def get_memory(session_id: str) -> Any:
    session = str(session_id).strip()
    if not session:
        raise ValueError("Missing session_id")
    return jsonify({"ok": True, "service": "voice_interaction_service", "session_id": session, "memory": ORCHESTRATOR.memory_store.summarize_context(session)})


@app.errorhandler(Exception)
def handle_exception(exc: Exception) -> Any:
    return jsonify({"ok": False, "error": type(exc).__name__, "message": str(exc)}), 500


if __name__ == "__main__":
    host = os.getenv("VOICE_HTTP_HOST", "0.0.0.0")
    port = int(os.getenv("VOICE_HTTP_PORT", "8083"))
    debug = getenv_bool("FLASK_DEBUG", False)
    app.run(host=host, port=port, debug=debug)
