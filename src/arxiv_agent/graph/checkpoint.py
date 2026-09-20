"""Durable state between processes.

The briefing run and the QA loop are usually separate invocations (often days
apart). Everything needed to resume lives in two places:

  ~/.arxiv_agent/sessions/<session_id>/state.json   <- the AgentState
  ~/.arxiv_agent/vectorstore/                       <- the embedded chunks

Writes are atomic (tmp file + os.replace) so a Ctrl-C mid-write cannot leave a
truncated session behind.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from ..state import AgentState


class FileCheckpointer:
    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        path = self.sessions_dir / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, state: AgentState) -> None:
        target = self.session_dir(state.session_id) / "state.json"
        payload = json.dumps(state.to_dict(), indent=2, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def load(self, session_id: str) -> AgentState:
        target = self.sessions_dir / session_id / "state.json"
        if not target.exists():
            raise FileNotFoundError(f"no such session: {session_id}")
        return AgentState.from_dict(json.loads(target.read_text(encoding="utf-8")))

    def latest(self) -> AgentState | None:
        sessions = self.list_sessions()
        return self.load(sessions[0]["session_id"]) if sessions else None

    def list_sessions(self) -> list[dict]:
        rows: list[dict] = []
        for path in self.sessions_dir.glob("*/state.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            paper = data.get("paper") or {}
            rows.append(
                {
                    "session_id": data.get("session_id", path.parent.name),
                    "updated_at": data.get("updated_at", 0),
                    "status": data.get("status", "?"),
                    "input": data.get("raw_input", ""),
                    "arxiv_id": paper.get("arxiv_id", ""),
                    "title": paper.get("title", ""),
                    "questions": len(data.get("qa_history", [])),
                }
            )
        return sorted(rows, key=lambda r: r["updated_at"], reverse=True)
