"""Raspberry Pi camera MJPEG server for the LifeSignal dashboard.

This service is intentionally independent from ``server.py``.  It uses the
Raspberry Pi OS ``picamera2`` package and exposes a small HTTP API:

* ``/stream.mjpg`` - multipart MJPEG stream for the dashboard
* ``/snapshot.jpg`` - one JPEG frame
* ``/healthz`` - basic liveness check

Install Picamera2 with the Raspberry Pi OS package manager rather than pip:
``sudo apt install -y python3-picamera2``.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final
from urllib.parse import urlsplit


BOUNDARY: Final[bytes] = b"frame"


class Camera:
    """Thread-safe Picamera2 JPEG capture wrapper."""

    def __init__(self, width: int, height: int) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:  # pragma: no cover - Raspberry Pi runtime
            raise RuntimeError(
                "Picamera2가 설치되어 있지 않습니다. "
                "'sudo apt install -y python3-picamera2'를 실행해주세요."
            ) from exc

        self._lock = threading.Lock()
        self._camera = Picamera2()
        configuration = self._camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._camera.configure(configuration)
        self._camera.start()
        # The first frame can be incomplete immediately after start().
        time.sleep(0.5)

    def capture_jpeg(self) -> bytes:
        with self._lock:
            output = io.BytesIO()
            self._camera.capture_file(output, format="jpeg")
            return output.getvalue()

    def close(self) -> None:
        with self._lock:
            self._camera.stop()
            self._camera.close()


class CameraRequestHandler(BaseHTTPRequestHandler):
    camera: Camera
    frame_interval: float

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = urlsplit(self.path).path
        if path == "/healthz":
            body = json.dumps({"ok": True}, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self._cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/snapshot.jpg":
            self._send_snapshot()
            return

        if path == "/stream.mjpg":
            self._send_stream()
            return

        body = (
            b"LifeSignal camera server\n"
            b"GET /stream.mjpg, /snapshot.jpg, or /healthz\n"
        )
        self.send_response(HTTPStatus.OK)
        self._cors_headers()
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_snapshot(self) -> None:
        try:
            frame = self.camera.capture_jpeg()
            self.send_response(HTTPStatus.OK)
            self._cors_headers()
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))

    def _send_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self._cors_headers()
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            while True:
                frame = self.camera.capture_jpeg()
                self.wfile.write(
                    b"--" + BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                    + frame
                    + b"\r\n"
                )
                self.wfile.flush()
                time.sleep(self.frame_interval)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def log_message(self, format: str, *args: object) -> None:
        # Keep camera frames out of the terminal; report only errors/startup.
        if args and str(args[0]).startswith("4"):
            super().log_message(format, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LifeSignal Raspberry Pi camera server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    if args.port < 1 or args.port > 65535:
        parser.error("포트는 1~65535 범위여야 합니다.")
    if args.width < 160 or args.height < 120:
        parser.error("카메라 해상도가 너무 작습니다.")
    if args.fps <= 0:
        parser.error("FPS는 0보다 커야 합니다.")
    return args


def main() -> None:
    args = parse_args()
    camera = Camera(args.width, args.height)
    handler = type(
        "ConfiguredCameraRequestHandler",
        (CameraRequestHandler,),
        {"camera": camera, "frame_interval": 1.0 / args.fps},
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"LifeSignal 카메라 서버 실행: http://{args.host}:{args.port} "
        f"({args.width}x{args.height}, {args.fps:g}fps)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n카메라 서버를 종료합니다.")
    finally:
        server.server_close()
        camera.close()


if __name__ == "__main__":
    main()
