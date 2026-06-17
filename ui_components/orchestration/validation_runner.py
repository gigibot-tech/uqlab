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


def render_local_validation_viz(
    *,
    key_prefix: str = "val_viz",
    show_fig3: bool = True,
    show_fig4: bool = True,
    architecture_label: str | None = None,
) -> None:
    """
    Paper-style Fig. 3 / Fig. 4 plots from ``results/validation/*/metrics.csv``.

    These plots are read from *local validation preset artifacts* on disk, not from
    the "Paper sweeps" you launch via the progressive app API. We still annotate the
    figure titles with the architecture label found in `metrics.csv`.
    """
    from uqlab.results_io import load_unified_metrics
    from uqlab.shared.notebook_utils.signals import resolve_x_col
    from uqlab.ui_components.visualization.signals.signal_diagnostic_viz import (
        plot_architecture_row_sweep,
    )

    st.markdown("#### 📈 Local sweep plots")
    head, refresh_col = st.columns([4, 1])
    with refresh_col:
        if st.button("🔄 Refresh plots", key=f"{key_prefix}_refresh", use_container_width=True):
            st.rerun()

    default_arch = architecture_label or "DINOv2 + MLP"
    panels: list[tuple[str, str, str]] = []
    if show_fig3:
        panels.append(("dataset_size", "Fig. 3 — Under-train (epistemic)", default_arch))
    if show_fig4:
        panels.append(("label_noise", "Fig. 4 — Label noise (aleatoric)", default_arch))
    any_data = False

    for sweep_type, title, architecture in panels:
        df = load_unified_metrics(
            sweep_type,
            sources=("pytorch_validation",),
            results_root=_PROJECT_ROOT / "results",
        )
        if df.empty:
            st.info(f"**{title}** — no rows in `metrics.csv` yet. Run the sweep above.")
            continue

        arch = architecture
        if "architecture" in df.columns:
            arch_df = df[df["architecture"] == arch]
            if arch_df.empty:
                archs = sorted(df["architecture"].dropna().unique().tolist())
                fallback_arch = archs[0] if archs else arch
                # If the caller asked for a specific arch, explain the fallback to avoid
                # confusion like "it says DINOv2 but I ran ResNet".
                if architecture_label is not None and fallback_arch != arch:
                    st.warning(
                        f"Plot requested architecture `{architecture_label}`, but `metrics.csv` "
                        f"only has `{fallback_arch}`. Falling back to `{fallback_arch}`."
                    )
                arch = fallback_arch
                arch_df = df[df["architecture"] == arch] if archs else df
            df = arch_df

        try:
            x_col = resolve_x_col(df, sweep_type)
        except ValueError:
            st.warning(f"**{title}** — could not resolve x-axis column.")
            continue

        if x_col not in df.columns and "sweep_value" in df.columns:
            x_col = "sweep_value"

        x_values = (
            sorted(df[x_col].dropna().unique().tolist()) if x_col in df.columns else []
        )
        
        # Always show title
        st.markdown(f"**{title}** · `{arch}`")
        
        if len(x_values) < 2:
            st.info(
                f"Need ≥2 sweep points for plot (have {len(x_values)}). "
                f"Showing experiment details below."
            )
        else:
            any_data = True
            highlight = x_values[-1]
            fig = plot_architecture_row_sweep(
                df,
                architecture=arch,
                x_col=x_col,
                highlight_x=highlight,
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_{sweep_type}")
            else:
                st.caption("No architecture-level metrics to plot.")
        
        # Always show experiment details in expander
        with st.expander(f"📊 Experiment Details ({len(df)} experiments)", expanded=False):
            # Select columns to display
            display_cols = []
            
            # Add experiment ID if available
            if "experiment_id" in df.columns:
                display_cols.append("experiment_id")
            
            # Add sweep parameter
            if x_col in df.columns:
                display_cols.append(x_col)
            
            # Add the SOURCE metrics used in the calculation (mutual_info_mean, predictive_entropy_mean)
            source_metrics = [c for c in df.columns if c in ["mutual_info_mean", "predictive_entropy_mean"]]
            display_cols.extend(source_metrics)
            
            # Add the PLOTTED metrics (mean_epistemic_uncertainty, mean_aleatoric_uncertainty)
            # These are what the blue/green lines show!
            plotted_metrics = [c for c in df.columns if any(
                m in c for m in ["mean_epistemic_uncertainty", "mean_aleatoric_uncertainty"]
            )]
            display_cols.extend(plotted_metrics)
            
            # Prioritize mutual_info AUROC columns (used in epistemic calculation)
            mutual_info_auroc = [c for c in df.columns if c.startswith("mutual_info_") and "auroc" in c.lower()]
            display_cols.extend(mutual_info_auroc)
            
            # Add all other AUROC columns (epistemic, aleatoric, baseline signals)
            auroc_cols = [c for c in df.columns if "auroc" in c.lower() and not c.startswith("mutual_info_")]
            display_cols.extend(auroc_cols)
            
            # Add accuracy
            if "accuracy" in df.columns:
                display_cols.append("accuracy")
            
            # Filter to available columns and remove duplicates
            display_cols = [c for c in display_cols if c in df.columns]
            display_cols = list(dict.fromkeys(display_cols))  # Remove duplicates while preserving order
            
            if display_cols:
                # Create a copy to add calculation column
                display_df = df[display_cols].copy()
                
                # Add calculation column showing the formula with actual values
                if "predictive_entropy_mean" in df.columns and "mutual_info_mean" in df.columns:
                    # Build formula strings row by row
                    calc_strings = []
                    for _, row in df.iterrows():
                        pred_ent = row["predictive_entropy_mean"]
                        mut_info = row["mutual_info_mean"]
                        result = pred_ent - mut_info
                        calc_strings.append(f"{pred_ent:.4f} − {mut_info:.4f} = {result:.4f}")
                    display_df["aleatoric_calc"] = calc_strings
                
                st.caption(
                    "💡 **Plot calculations:** "
                    "Green = `mean_epistemic_uncertainty` = `mutual_info_mean` | "
                    "Blue = `mean_aleatoric_uncertainty` = `predictive_entropy_mean` − `mutual_info_mean` "
                    "(see `aleatoric_calc` column for formula)"
                )
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                # Fallback: show all columns
                st.dataframe(df, use_container_width=True, hide_index=True)

    if any_data:
        st.caption(
            "Local validation preset plots read from `results/validation/*_sweep/metrics.csv`. "
            "These are for preset sweeps (not the API paper-sweep campaigns)."
        )


def render_preset_validation_sweeps(
    *,
    key_prefix: str = "val",
    show_viz: bool = True,
) -> str:
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
    if show_viz:
        render_local_validation_viz(key_prefix=f"{key_prefix}_viz")
    return mode
