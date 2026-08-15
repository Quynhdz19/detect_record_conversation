"""In-memory conversation sessions for Pi / API clients."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

_LOCK = threading.Lock()
_SESSIONS: dict[str, dict[str, Any]] = {}
_MAX_TURNS = 200


def create_session(device_id: str = "") -> dict[str, Any]:
    sid = uuid.uuid4().hex[:12]
    now = time.time()
    session = {
        "session_id": sid,
        "device_id": device_id,
        "created_at": now,
        "updated_at": now,
        "turns": [],
    }
    with _LOCK:
        _SESSIONS[sid] = session
    return snapshot(session)


def get_session(session_id: str) -> dict[str, Any] | None:
    with _LOCK:
        sess = _SESSIONS.get(session_id)
        return snapshot(sess) if sess else None


def append_turn(session_id: str, text: str, used_tse: bool = False) -> dict[str, Any] | None:
    if not text:
        return get_session(session_id)
    with _LOCK:
        sess = _SESSIONS.get(session_id)
        if sess is None:
            return None
        last = sess["turns"][-1]["text"].lower() if sess["turns"] else ""
        if text.lower() == last:
            sess["updated_at"] = time.time()
            return snapshot(sess)
        sess["turns"].append(
            {
                "index": len(sess["turns"]) + 1,
                "ts": time.time(),
                "text": text,
                "used_tse": used_tse,
            }
        )
        if len(sess["turns"]) > _MAX_TURNS:
            sess["turns"] = sess["turns"][-_MAX_TURNS:]
        sess["updated_at"] = time.time()
        return snapshot(sess)


def conversation_text(session: dict[str, Any]) -> str:
    return "\n".join(t["text"] for t in session.get("turns", []))


def snapshot(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session["session_id"],
        "device_id": session.get("device_id") or "",
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "turns": list(session["turns"]),
        "conversation": conversation_text(session),
    }
