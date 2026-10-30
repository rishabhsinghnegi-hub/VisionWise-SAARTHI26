"""One persistent ESP32 MJPEG reader shared by inference and live preview."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterator

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))

ESP32_STREAM_URL = os.getenv("ESP32_STREAM_URL")
MAX_FRAME_BYTES = 8 * 1024 * 1024
FRAME_MAX_AGE_SECONDS = 5.0
logger = logging.getLogger("visionwise.camera")


class Esp32CameraStream:
    def __init__(self, url: str | None):
        self.url = url
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._response: httpx.Response | None = None
        self._frame: bytes | None = None
        self._frame_id = 0
        self._frame_at = 0.0
        self._last_error: str | None = None

    def _ensure_started(self) -> None:
        if not self.url:
            raise RuntimeError("ESP32_STREAM_URL is missing from apps/api/.env")
        while True:
            with self._condition:
                thread = self._thread
                if thread is None or not thread.is_alive():
                    self._stop_event.clear()
                    self._thread = threading.Thread(target=self._read_loop, name="esp32-mjpeg-reader", daemon=True)
                    self._thread.start()
                    return
                if not self._stop_event.is_set():
                    return
            thread.join(timeout=0.1)

    def _publish(self, frame: bytes) -> None:
        with self._condition:
            self._frame = frame
            self._frame_id += 1
            self._frame_at = time.monotonic()
            self._last_error = None
            self._condition.notify_all()

    def _read_loop(self) -> None:
        assert self.url is not None
        timeout = httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0)
        while not self._stop_event.is_set():
            try:
                with httpx.stream("GET", self.url, timeout=timeout, follow_redirects=True) as response:
                    with self._condition:
                        self._response = response
                    if self._stop_event.is_set():
                        break
                    response.raise_for_status()
                    buffer = bytearray()
                    for chunk in response.iter_bytes():
                        if self._stop_event.is_set():
                            break
                        buffer.extend(chunk)
                        while True:
                            start = buffer.find(b"\xff\xd8")
                            if start < 0:
                                if len(buffer) > 1:
                                    del buffer[:-1]
                                break
                            end = buffer.find(b"\xff\xd9", start + 2)
                            if end < 0:
                                if start:
                                    del buffer[:start]
                                if len(buffer) > MAX_FRAME_BYTES:
                                    raise RuntimeError("ESP32-CAM JPEG frame exceeds the 8 MB limit")
                                break
                            frame = bytes(buffer[start:end + 2])
                            del buffer[:end + 2]
                            if len(frame) > MAX_FRAME_BYTES:
                                raise RuntimeError("ESP32-CAM JPEG frame exceeds the 8 MB limit")
                            self._publish(frame)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                message = f"{type(exc).__name__}: {exc}"
                with self._condition:
                    self._last_error = message
                    self._condition.notify_all()
                logger.warning("ESP32-CAM stream disconnected: %s", message)
                self._stop_event.wait(0.75)
            finally:
                with self._condition:
                    self._response = None

    def get_frame(self, timeout: float = 15.0, after_frame_id: int = 0) -> tuple[bytes, int]:
        self._ensure_started()
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._stop_event.is_set():
                    raise RuntimeError("ESP32-CAM stream was stopped")
                fresh = self._frame is not None and time.monotonic() - self._frame_at <= FRAME_MAX_AGE_SECONDS
                if fresh and self._frame_id > after_frame_id:
                    return self._frame, self._frame_id
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = self._last_error or "No fresh JPEG frame received from ESP32-CAM."
                    raise RuntimeError(detail)
                self._condition.wait(min(remaining, 1.0))

    def iter_frames(self) -> Iterator[bytes]:
        frame_id = 0
        while not self._stop_event.is_set():
            try:
                frame, frame_id = self.get_frame(timeout=15.0, after_frame_id=frame_id)
            except RuntimeError as exc:
                if self._stop_event.is_set():
                    return
                logger.warning("Waiting for ESP32-CAM preview frame: %s", exc)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                + frame
                + b"\r\n"
            )

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            response = self._response
            self._condition.notify_all()
        if response is not None:
            response.close()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._condition:
            self._response = None
            self._frame = None
            self._frame_id = 0
            self._frame_at = 0.0
            if thread is None or not thread.is_alive():
                self._thread = None


camera_stream = Esp32CameraStream(ESP32_STREAM_URL)


def capture_frame() -> bytes:
    frame, _ = camera_stream.get_frame()
    return frame


def iter_preview_frames() -> Iterator[bytes]:
    return camera_stream.iter_frames()


def stop_camera_stream() -> None:
    camera_stream.stop()
