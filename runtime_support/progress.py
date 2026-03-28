from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import sys
from typing import Any, TextIO

PROGRESS_EVENT_PREFIX = "__NF_PROGRESS__"


def progress_bar(completed: int, total: int, *, width: int = 18) -> str:
    if total <= 0:
        total = 1
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = int(round(width * ratio))
    pct = int(round(ratio * 100))
    return f"[{'#' * filled}{'-' * (width - filled)}] {completed}/{total} {pct:3d}%"


@dataclass
class ModelProgressState:
    job_name: str
    model_index: int
    total_models: int
    total_steps: int
    completed_steps: int = 0
    total_folds: int | None = None
    current_fold: int | None = None
    phase: str | None = None
    status: str = "queued"
    detail: str | None = None
    event: str = "queued"

    @property
    def progress_pct(self) -> int:
        total = max(self.total_steps, 1)
        return int(round(min(max(self.completed_steps / total, 0.0), 1.0) * 100))

    def progress_text(self) -> str:
        return progress_bar(self.completed_steps, self.total_steps)

    def to_event_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["progress_pct"] = self.progress_pct
        payload["progress_text"] = self.progress_text()
        return payload


def emit_progress_event(state: ModelProgressState, stream: TextIO | None = None) -> None:
    target = stream or sys.stdout
    target.write(
        f"{PROGRESS_EVENT_PREFIX}{json.dumps(state.to_event_payload(), ensure_ascii=False)}\n"
    )
    target.flush()


def parse_progress_event(line: str) -> dict[str, Any] | None:
    if not line.startswith(PROGRESS_EVENT_PREFIX):
        return None
    return json.loads(line[len(PROGRESS_EVENT_PREFIX) :])


def build_summary_line(states: list[ModelProgressState]) -> str:
    total = len(states)
    completed = sum(state.status == "completed" for state in states)
    failed = sum(state.status == "failed" for state in states)
    queued = sum(state.status == "queued" for state in states)
    running = [
        f"{state.job_name}({state.model_index}/{state.total_models})"
        for state in states
        if state.status == "running"
    ]
    current = ",".join(running) if running else "-"
    return (
        "[summary] "
        f"completed={completed}/{total} failed={failed} queued={queued} current={current}"
    )


def build_model_line(state: ModelProgressState) -> str:
    parts = [
        f"[model:{state.job_name}]",
        f"idx={state.model_index}/{state.total_models}",
        f"status={state.status}",
    ]
    if state.phase:
        parts.append(f"phase={state.phase}")
    if state.current_fold is not None:
        fold_text = f"fold={state.current_fold + 1}"
        if state.total_folds is not None:
            fold_text += f"/{state.total_folds}"
        parts.append(fold_text)
    parts.append(state.progress_text())
    if state.detail:
        parts.append(state.detail)
    return " ".join(parts)


class ConsoleProgressRenderer:
    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        enable_ansi: bool | None = None,
    ) -> None:
        self.stream = stream or sys.stdout
        if enable_ansi is None:
            enable_ansi = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enable_ansi = enable_ansi
        self._rendered_lines = 0

    def render(self, states: list[ModelProgressState]) -> None:
        lines = [build_summary_line(states), *[build_model_line(state) for state in states]]
        if self.enable_ansi:
            if self._rendered_lines:
                self.stream.write(f"\x1b[{self._rendered_lines}F")
            for idx, line in enumerate(lines):
                self.stream.write("\r\x1b[2K" + line)
                if idx < len(lines) - 1:
                    self.stream.write("\n")
            self.stream.flush()
            self._rendered_lines = len(lines)
            return
        self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()

    def close(self) -> None:
        if self.enable_ansi and self._rendered_lines:
            self.stream.write("\n")
            self.stream.flush()
            self._rendered_lines = 0
