"""Host-side time per stage of an engine step, for finding CPU overhead between GPU steps.

Enabled with ``TANDEM_STEP_TIMING=<N>``: every N steps the mean microseconds per step of each
stage are logged. Disabled, every call is a no-op. Stages are marked in order: each ``mark``
charges the time since the previous mark (or ``start``) to its name.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict

logger = logging.getLogger("tandem.step_timer")


class _StepTimer:
    def __init__(self, every: int) -> None:
        self.every = every
        self.acc: dict[str, float] = defaultdict(float)
        self.steps = 0
        self.t = 0.0

    def start(self) -> None:
        self.t = time.perf_counter()

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.acc[name] += now - self.t
        self.t = now

    def end_step(self) -> None:
        self.steps += 1
        if self.steps % self.every == 0:
            total = sum(self.acc.values())
            parts = "  ".join(f"{k} {v / self.every * 1e6:.0f}" for k, v in self.acc.items())
            logger.info("host us/step over %d steps: total %.0f | %s", self.every, total / self.every * 1e6, parts)
            self.acc.clear()


class _NoTimer:
    def start(self) -> None:
        pass

    def mark(self, name: str) -> None:
        pass

    def end_step(self) -> None:
        pass


_every = int(os.environ.get("TANDEM_STEP_TIMING", "0") or 0)
TIMER = _StepTimer(_every) if _every > 0 else _NoTimer()


class _Trace:
    """``TANDEM_TRACE=<path>``: one JSON line per engine event (request received by the core,
    step launched with its composition, outputs sent), ``t`` = time.perf_counter() in ms, the
    clock of benchmarks/bench_serving.py's per-request dumps (CLOCK_MONOTONIC on Linux)."""

    def __init__(self, path: str) -> None:
        self.f = open(path, "a", buffering=1, encoding="utf-8")

    def __call__(self, ev: str, **kw) -> None:
        import json

        self.f.write(json.dumps({"t": round(time.perf_counter() * 1e3, 3), "ev": ev, **kw}) + "\n")


_trace_path = os.environ.get("TANDEM_TRACE", "")
TRACE = _Trace(_trace_path) if _trace_path else None
