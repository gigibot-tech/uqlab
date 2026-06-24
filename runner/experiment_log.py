"""Capture stdout/stderr for one experiment run into ``results/experiment.log``."""

from __future__ import annotations

import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

EXPERIMENT_LOG_FILENAME = "experiment.log"


class _TeeStream:
    """Write to a log file and the original stream (tqdm, print, warnings)."""

    def __init__(self, original, tee_target) -> None:
        self._original = original
        self._tee = tee_target

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._original.write(data)
        self._tee.write(data)
        self._tee.flush()
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._tee.flush()

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return getattr(self._original, "isatty", lambda: False)()

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")


def experiment_log_path(results_dir: Path) -> Path:
    """Path to the full run log under a results directory."""
    return Path(results_dir) / EXPERIMENT_LOG_FILENAME


def read_experiment_log(
    results_dir: Path,
    *,
    tail_chars: int | None = 32_000,
) -> str | None:
    """Return log text, optionally truncated to the last *tail_chars* characters."""
    path = experiment_log_path(results_dir)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if tail_chars is not None and len(text) > tail_chars:
        return text[-tail_chars:]
    return text


def infer_experiment_id(*, results_dir: Path, config_path: Path | None = None) -> str | None:
    """Best-effort experiment UUID from ``…/<id>/results`` or ``…/<id>/config.yaml``."""
    results_dir = Path(results_dir)
    if results_dir.name == "results":
        parent = results_dir.parent
        if (parent / "config.yaml").is_file():
            return parent.name
    if config_path is not None:
        parent = Path(config_path).parent
        if (parent / "config.yaml").is_file():
            return parent.name
    return None


@contextmanager
def capture_experiment_log(
    results_dir: Path,
    *,
    experiment_id: str | None = None,
    config_path: Path | None = None,
) -> Iterator[Path]:
    """
    Tee stdout/stderr to ``results_dir/experiment.log`` for the duration.

    Each invocation appends a new section (run start banner → output → status footer).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiment_log_path(results_dir)
    run_id = experiment_id or infer_experiment_id(
        results_dir=results_dir, config_path=config_path
    )
    started = datetime.now(timezone.utc)

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"EXPERIMENT LOG — started {started.isoformat()}\n")
        if run_id:
            log_file.write(f"Experiment ID: {run_id}\n")
        if config_path is not None:
            log_file.write(f"Config: {Path(config_path).resolve()}\n")
        log_file.write(f"Results directory: {results_dir.resolve()}\n")
        log_file.write("=" * 80 + "\n")
        log_file.flush()

        out_tee = _TeeStream(sys.stdout, log_file)
        err_tee = _TeeStream(sys.stderr, log_file)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out_tee, err_tee  # type: ignore[assignment]
        status = "completed"
        try:
            yield log_path
        except BaseException:
            status = "failed"
            log_file.write("\n")
            log_file.write(traceback.format_exc())
            log_file.flush()
            raise
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            ended = datetime.now(timezone.utc)
            log_file.write("\n")
            log_file.write("=" * 80 + "\n")
            log_file.write(f"EXPERIMENT LOG — {status} {ended.isoformat()}\n")
            log_file.write("=" * 80 + "\n")
            log_file.flush()
