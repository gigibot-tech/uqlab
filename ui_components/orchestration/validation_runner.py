"""
Shared UI + subprocess helpers for preset validation sweeps.

Used by Hypothesis Validation and Custom experiments (Unified Builder) so both
tabs launch the same ``run_validation_experiments.py`` runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import streamlit as st


def _resolve_walaris_cen_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").is_file() and (p / "scripts").is_dir():
            return p
    return here.parents[3]


_PROJECT_ROOT = _resolve_walaris_cen_root()
_SRC = _PROJECT_ROOT / "src"
for _entry in (str(_SRC), str(_PROJECT_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from uqlab_orchestrator.config.validation_config import ARCHITECTURES, DATASET_SIZE_SWEEP, LABEL_NOISE_SWEEP

_N_ARCH = len(ARCHITECTURES)
_ARCH_NAMES = ", ".join(a["name"] for a in ARCHITECTURES.values())


def _subprocess_env() -> dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    paths = [str(_SRC), str(_PROJECT_ROOT)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _stream_subprocess(
    cmd: list[str],
    cwd: Path,
    on_line: Callable[[str], None],
    *,
    timeout: int = 3600,
    max_buffered_lines: int = 4000,
) -> tuple[int, list[str]]:
    import time

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=_subprocess_env(),
    )
    buffered: deque[str] = deque(maxlen=max_buffered_lines)
    start = time.time()
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip()
            buffered.append(line)
            try:
                on_line(line)
            except Exception:
                pass
            if time.time() - start > timeout:
                process.terminate()
                buffered.append(f"[timeout after {timeout}s — process terminated]")
                break
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        buffered.append(f"[timeout after {timeout}s — process killed]")
    finally:
        if process.stdout is not None:
            try:
                process.stdout.close()
            except Exception:
                pass
    return process.returncode if process.returncode is not None else -1, list(buffered)


def run_validation_experiments(
    sweep_type: str,
    mode: str = "quick",
    *,
    on_line: Optional[Callable[[str], None]] = None,
    timeout: int = 3600,
) -> tuple[bool, str]:
    """Run ``scripts/run_validation_experiments.py`` (streaming optional)."""
    script_path = _PROJECT_ROOT / "scripts" / "run_validation_experiments.py"
    if not script_path.is_file():
        return False, f"Runner not found: {script_path}"

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--sweep",
        sweep_type,
        "--mode",
        mode,
    ]
    sink: list[str] = []

    def _record(line: str) -> None:
        sink.append(line)
        if on_line is not None:
            on_line(line)

    code, _ = _stream_subprocess(cmd, cwd=_PROJECT_ROOT, on_line=_record, timeout=timeout)
    return code == 0, "\n".join(sink)


def _execute_sweep_ui(sweep: str, label: str, mode: str, *, key_prefix: str) -> None:
    status = st.status(
        f"Running {label} sweep (mode={mode})…",
        expanded=True,
        state="running",
    )
    with status:
        log_placeholder = st.empty()
        tail: deque[str] = deque(maxlen=200)

        def _on_line(line: str) -> None:
            tail.append(line)
            log_placeholder.code("\n".join(tail), language="text")

        success, full_output = run_validation_experiments(sweep, mode, on_line=_on_line)
        if success:
            status.update(
                label=f"Done: {label} (mode={mode})",
                state="complete",
                expanded=False,
            )
            with st.expander("Full output"):
                st.code(full_output, language="text")
        else:
            status.update(label=f"Failed: {label}", state="error", expanded=True)
            with st.expander("Full output"):
                st.code(full_output, language="text")


def render_preset_validation_sweeps(*, key_prefix: str = "val") -> str:
    """
    Render quick/full mode + dataset-size / label-noise run buttons.

    Returns the selected mode (``quick`` or ``full``).
    """
    mode = st.radio(
        "Run mode",
        ["quick", "full"],
        index=0,
        horizontal=True,
        key=f"{key_prefix}_mode",
        help=(
            f"Quick: dataset sizes {DATASET_SIZE_SWEEP['quick']}, "
            f"noise % {LABEL_NOISE_SWEEP['quick']}. "
            f"Full: {DATASET_SIZE_SWEEP['full']} / {LABEL_NOISE_SWEEP['full']}."
        ),
    )

    col1, col2 = st.columns(2)
    with col1:
        st.caption(
            f"Dataset size sweep (epistemic) — {_N_ARCH} architecture(s): {_ARCH_NAMES}; noise 0%"
        )
        run_epis = st.button(
            "Run dataset size sweep",
            key=f"{key_prefix}_run_dataset_size",
            type="primary",
            use_container_width=True,
        )
    with col2:
        st.caption(
            f"Label noise sweep (aleatoric) — {_N_ARCH} architecture(s): {_ARCH_NAMES}; fixed train size"
        )
        run_alea = st.button(
            "Run label noise sweep",
            key=f"{key_prefix}_run_label_noise",
            use_container_width=True,
        )

    if run_epis:
        _execute_sweep_ui("dataset_size", "dataset size", mode, key_prefix=key_prefix)
    if run_alea:
        _execute_sweep_ui("label_noise", "label noise", mode, key_prefix=key_prefix)

    st.caption(
        "Output: `results/validation/<sweep>_sweep/<arch>_*/` with "
        "`summary.json`, `signal_formulas.json`, `per_sample_signals.csv`, "
        "`results.pt` → merged into `metrics.csv`."
    )
    return mode
