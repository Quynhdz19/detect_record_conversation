#!/usr/bin/env python3
"""Pi client: stream camera + mic qua WebSocket, in text hội thoại.

    pip install websocket-client sounddevice opencv-python-headless numpy
    python clients/pi_client.py --url http://SERVER:8000

Gửi:
  0x01 + JPEG
  0x02 + PCM16LE mono 16kHz
Nhận:
  {"type":"transcript","text":"...","conversation":"..."}
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import urllib.parse

from websocket import ABNF, WebSocketConnectionClosedException, create_connection

TAG_JPEG = b"\x01"
TAG_PCM = b"\x02"


def to_ws_url(base: str, device_id: str, token: str, session_id: str) -> str:
    ws = base.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
    q = {"device_id": device_id}
    if token:
        q["token"] = token
    if session_id:
        q["session_id"] = session_id
    return f"{ws}/api/pi/ws?{urllib.parse.urlencode(q)}"


class PiStreamer:
    def __init__(self, url: str, camera: int, fps: float, jpeg_quality: int):
        self.url = url
        self.camera = camera
        self.frame_interval = 1.0 / max(fps, 1.0)
        self.jpeg_quality = jpeg_quality
        self.out_q: queue.Queue[bytes | str] = queue.Queue(maxsize=80)
        self.stop = threading.Event()
        self.session_id = ""
        self.ws = None

    def _send_loop(self) -> None:
        while not self.stop.is_set():
            try:
                item = self.out_q.get(timeout=0.2)
            except queue.Empty:
                continue
            ws = self.ws
            if ws is None:
                continue
            try:
                if isinstance(item, str):
                    ws.send(item)
                else:
                    ws.send(item, opcode=ABNF.OPCODE_BINARY)
            except Exception:
                self.stop.set()
                return

    def _audio_loop(self) -> None:
        import numpy as np
        import sounddevice as sd

        def cb(indata, frames, time_info, status):
            if self.stop.is_set():
                return
            pcm = np.ascontiguousarray(indata[:, 0]).tobytes()
            try:
                self.out_q.put_nowait(TAG_PCM + pcm)
            except queue.Full:
                pass

        with sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="int16",
            blocksize=4096,
            callback=cb,
        ):
            while not self.stop.is_set():
                time.sleep(0.1)

    def _video_loop(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            print("Không mở được camera", self.camera)
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                )
                if ok:
                    try:
                        self.out_q.put_nowait(TAG_JPEG + buf.tobytes())
                    except queue.Full:
                        pass
                time.sleep(self.frame_interval)
        finally:
            cap.release()

    def _recv_loop(self) -> None:
        while not self.stop.is_set():
            ws = self.ws
            if ws is None:
                time.sleep(0.05)
                continue
            try:
                ws.settimeout(0.4)
                msg = ws.recv()
            except WebSocketConnectionClosedException:
                self.stop.set()
                return
            except Exception:
                continue
            if not isinstance(msg, str):
                continue
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            kind = data.get("type")
            if kind == "ready":
                self.session_id = data.get("session_id") or self.session_id
                print("WS ready session=", self.session_id)
            elif kind == "transcript":
                text = data.get("text") or ""
                if data.get("final"):
                    print(">>", text)
                    conv = data.get("conversation") or ""
                    if conv:
                        print("--- hội thoại ---\n" + conv + "\n")
                else:
                    print("\r…", text, end="   ", flush=True)
            elif kind == "error":
                print("server error:", data.get("text"))
            elif kind == "ping":
                try:
                    self.out_q.put_nowait(json.dumps({"type": "pong"}))
                except queue.Full:
                    pass

    def run_once(self) -> str:
        print("Connecting", self.url)
        self.ws = create_connection(self.url, timeout=20)
        self.stop.clear()
        threads = [
            threading.Thread(target=self._send_loop, daemon=True),
            threading.Thread(target=self._audio_loop, daemon=True),
            threading.Thread(target=self._video_loop, daemon=True),
            threading.Thread(target=self._recv_loop, daemon=True),
        ]
        for t in threads:
            t.start()
        try:
            while not self.stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            try:
                self.ws.send(json.dumps({"type": "flush"}))
                time.sleep(0.3)
            except Exception:
                pass
            raise
        finally:
            self.stop.set()
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
        return self.session_id


def main() -> None:
    p = argparse.ArgumentParser(description="Stream cam+mic tới server qua WebSocket")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="http(s) origin của server")
    p.add_argument("--device-id", default="pi")
    p.add_argument("--session-id", default="", help="reconnect cùng hội thoại")
    p.add_argument("--token", default="")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--reconnect", action="store_true", default=True)
    args = p.parse_args()

    session_id = args.session_id
    print("WebSocket stream. Ctrl+C để dừng.")
    while True:
        ws_url = to_ws_url(args.url, args.device_id, args.token, session_id)
        streamer = PiStreamer(ws_url, args.camera, args.fps, args.jpeg_quality)
        streamer.session_id = session_id
        try:
            session_id = streamer.run_once() or session_id
            break
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as exc:
            print("mất kết nối:", exc)
            if not args.reconnect:
                raise
            time.sleep(2)
            print("reconnect session=", session_id or "(new)")


if __name__ == "__main__":
    main()
