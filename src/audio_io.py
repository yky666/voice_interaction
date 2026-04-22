from __future__ import annotations

import base64
import math
import os
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common_utils import getenv_bool, getenv_int


def _env_text(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


@dataclass(slots=True)
class AudioIOConfig:
    input_device: str
    output_device: str
    sample_rate: int
    channels: int
    chunk_ms: int
    vad_threshold: float
    vad_start_chunks: int
    vad_end_silence_sec: float
    max_record_sec: float
    enable_barge_in: bool
    playback_backend: str

    @classmethod
    def from_env(cls) -> "AudioIOConfig":
        return cls(
            input_device=_env_text("VOICE_AUDIO_INPUT_DEVICE", "default"),
            output_device=_env_text("VOICE_AUDIO_OUTPUT_DEVICE", "default"),
            sample_rate=getenv_int("VOICE_AUDIO_SAMPLE_RATE", 16000),
            channels=getenv_int("VOICE_AUDIO_CHANNELS", 1),
            chunk_ms=getenv_int("VOICE_AUDIO_CHUNK_MS", 30),
            vad_threshold=float(_env_text("VOICE_VAD_THRESHOLD", "700")),
            vad_start_chunks=getenv_int("VOICE_VAD_START_CHUNKS", 3),
            vad_end_silence_sec=float(_env_text("VOICE_VAD_END_SILENCE_SEC", "1.0")),
            max_record_sec=float(_env_text("VOICE_MAX_RECORD_SEC", "12")),
            enable_barge_in=getenv_bool("VOICE_ENABLE_BARGE_IN", True),
            playback_backend=_env_text("VOICE_PLAYBACK_BACKEND", "aplay"),
        )


class AudioIO:
    def __init__(self, cfg: AudioIOConfig) -> None:
        self.cfg = cfg

    def capture_until_silence(self) -> dict[str, Any]:
        chunk_frames = int(self.cfg.sample_rate * self.cfg.chunk_ms / 1000)
        bytes_per_sample = 2
        chunk_size = chunk_frames * self.cfg.channels * bytes_per_sample
        max_chunks = max(1, int(self.cfg.max_record_sec * 1000 / self.cfg.chunk_ms))
        end_silence_chunks = max(1, int(self.cfg.vad_end_silence_sec * 1000 / self.cfg.chunk_ms))

        cmd = [
            "arecord",
            "-q",
            "-D",
            self.cfg.input_device,
            "-f",
            "S16_LE",
            "-r",
            str(self.cfg.sample_rate),
            "-c",
            str(self.cfg.channels),
            "-t",
            "raw",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if proc.stdout is None:
            raise RuntimeError("Failed to start arecord stdout pipe.")

        captured = bytearray()
        energies: list[float] = []
        speech_started = False
        speech_chunks = 0
        silence_chunks = 0
        try:
            for _ in range(max_chunks):
                chunk = proc.stdout.read(chunk_size)
                if not chunk:
                    break
                energy = _pcm_rms(chunk)
                energies.append(energy)
                if energy >= self.cfg.vad_threshold:
                    speech_chunks += 1
                else:
                    speech_chunks = 0

                if not speech_started:
                    if speech_chunks >= self.cfg.vad_start_chunks:
                        speech_started = True
                        captured.extend(chunk)
                else:
                    captured.extend(chunk)
                    if energy < self.cfg.vad_threshold:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0
                    if silence_chunks >= end_silence_chunks:
                        break
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()

        wav_bytes = _pcm_to_wav_bytes(bytes(captured), self.cfg.sample_rate, self.cfg.channels)
        return {
            "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
            "speech_detected": bool(speech_started and captured),
            "rms_max": max(energies) if energies else 0.0,
            "duration_sec": len(captured) / max(1, self.cfg.sample_rate * self.cfg.channels * bytes_per_sample),
        }

    def play_audio_b64_with_barge_in(self, audio_b64: str) -> dict[str, Any]:
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        return self.play_file_with_barge_in(tmp_path, delete_after=True)

    def play_file_with_barge_in(self, audio_path: str, delete_after: bool = False) -> dict[str, Any]:
        proc = subprocess.Popen(self._playback_command(audio_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stop_event = threading.Event()
        watcher: threading.Thread | None = None
        if self.cfg.enable_barge_in:
            watcher = threading.Thread(target=self._watch_for_barge_in, args=(proc, stop_event), daemon=True)
            watcher.start()
        try:
            proc.wait()
        finally:
            stop_event.set()
            if watcher is not None:
                watcher.join(timeout=1.0)
            if delete_after:
                Path(audio_path).unlink(missing_ok=True)
        return {"ok": True, "interrupted": proc.returncode not in (0, None), "backend": self.cfg.playback_backend}

    def _playback_command(self, audio_path: str) -> list[str]:
        if self.cfg.playback_backend == "pw-play":
            return ["pw-play", "--target", self.cfg.output_device, audio_path]
        if self.cfg.playback_backend == "ffplay":
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path]
        return ["aplay", "-q", "-D", self.cfg.output_device, audio_path]

    def _watch_for_barge_in(self, playback_proc: subprocess.Popen[Any], stop_event: threading.Event) -> None:
        chunk_frames = int(self.cfg.sample_rate * self.cfg.chunk_ms / 1000)
        chunk_size = chunk_frames * self.cfg.channels * 2
        cmd = [
            "arecord",
            "-q",
            "-D",
            self.cfg.input_device,
            "-f",
            "S16_LE",
            "-r",
            str(self.cfg.sample_rate),
            "-c",
            str(self.cfg.channels),
            "-t",
            "raw",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if proc.stdout is None:
            return
        speech_hits = 0
        try:
            while not stop_event.is_set() and playback_proc.poll() is None:
                chunk = proc.stdout.read(chunk_size)
                if not chunk:
                    time.sleep(0.02)
                    continue
                if _pcm_rms(chunk) >= self.cfg.vad_threshold:
                    speech_hits += 1
                else:
                    speech_hits = max(0, speech_hits - 1)
                if speech_hits >= self.cfg.vad_start_chunks:
                    playback_proc.terminate()
                    break
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()


def _pcm_rms(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    count = len(pcm_bytes) // 2
    if count <= 0:
        return 0.0
    total = 0.0
    for idx in range(0, len(pcm_bytes) - 1, 2):
        sample = int.from_bytes(pcm_bytes[idx : idx + 2], byteorder="little", signed=True)
        total += float(sample * sample)
    return math.sqrt(total / count)


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int, channels: int) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with wave.open(tmp_path, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
