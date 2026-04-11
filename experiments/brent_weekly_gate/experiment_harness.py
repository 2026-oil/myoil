from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


class ExperimentHarness:
    def __init__(self, time_budget: int = 300, results_path: str | Path = "results.json"):
        self.time_budget = float(time_budget)
        self.results_path = Path(results_path)
        self.started_at = time.perf_counter()
        self.metrics: dict[str, Any] = {}
        self.metric_history: list[tuple[str, float]] = []

    def should_stop(self) -> bool:
        return (time.perf_counter() - self.started_at) >= 0.8 * self.time_budget

    def check_value(self, value: Any, name: str) -> bool:
        try:
            v = float(value)
        except Exception:
            return False
        return bool(np.isfinite(v))

    def report_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = float(value)
        self.metric_history.append((name, float(value)))

    def finalize(self) -> None:
        payload = {
            "elapsed_seconds": float(time.perf_counter() - self.started_at),
            "time_budget_seconds": float(self.time_budget),
            "metrics": self.metrics,
            "metric_history": self.metric_history,
        }
        with self.results_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
