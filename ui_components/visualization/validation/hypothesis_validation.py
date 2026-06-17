"""
Hypothesis Validation UI Component for Streamlit Dashboard.

Provides an interactive interface to run validation experiments and analyze
whether uncertainty signals properly disentangle epistemic and aleatoric uncertainty.
"""

import json
import os
import streamlit as st
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _resolve_uqlab_cen_root() -> Path:
    """
    Repo root (contains ``pyproject.toml`` and ``scripts/``).

    Do not rely on a fixed ``parents[N]`` depth: ``__file__`` depth varies with
    symlink layout, editable installs, and refactors, which previously produced
    bogus paths like ``src/uqlab/analyze_validation_results.py``.
    """
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").is_file() and (p / "scripts").is_dir():
            return p
    return here.parents[3]


_PROJECT_ROOT = _resolve_uqlab_cen_root()
_SRC = _PROJECT_ROOT / "src"
for _p in (_SRC, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _analysis_script_path() -> Path:
    """Primary location after restructure; optional legacy root copy."""
    legacy = _PROJECT_ROOT / "scripts" / "legacy" / "analyze_validation_results.py"
    if legacy.is_file():
        return legacy
    root_copy = _PROJECT_ROOT / "analyze_validation_results.py"
    if root_copy.is_file():
        return root_copy
    return legacy

from uqlab.notebook_support.metric_specs import AUROC_ONLY, UNCERTAINTY_DECOMPOSITION
from uqlab.notebook_support.method_comparison_plotly import (
    create_method_uncertainty_comparison_figure,
)
from uqlab.notebook_support.signals import (
    ALEATORIC_SIGNALS,
    DISENTANGLEMENT_LABELS,
    EPISTEMIC_SIGNALS,
    ROW3_CANDIDATE_SIGNALS,
    SIGNAL_LABELS,
    SIGNAL_NAMES,
    disentanglement_label,
    get_row3_signals,
    get_top_n_signals,
    present_architectures,
    present_datasets,
    present_disentanglements,
    present_sources,
    resolve_x_col,
    sweep_to_auroc_type,
)
from uqlab.results_io import DATASET_LABELS, DATASETS, dataset_label, load_unified_metrics
from uqlab.run_artifacts import load_per_sample_table, load_run_directory
from uqlab.ui_components.visualization.signals.per_sample_signals_viz import (
    render_per_sample_signal_visualizations,
)
from uqlab.run_artifacts import FAST_PILOT_SIGNAL_NAMES
from uqlab.ui_components.visualization.signals.signal_diagnostic_viz import render_visualize_main
from uqlab.ui_components.orchestration.validation_runner import (
    render_preset_validation_sweeps,
    run_validation_experiments,
)

# Human-readable labels for the ``source`` key column.
_SOURCE_LABELS = {
    "pytorch_validation": "Your PyTorch attribution",
    "paper_keras": "Paper Keras (reference)",
}


def _source_label(key: str) -> str:
    return _SOURCE_LABELS.get(key, key.replace("_", " ").title())

__all__ = [
    "SIGNAL_NAMES",
    "EPISTEMIC_SIGNALS",
    "ALEATORIC_SIGNALS",
    "ROW3_CANDIDATE_SIGNALS",
    "SIGNAL_LABELS",
    "get_row3_signals",
    "get_top_n_signals",
    "create_method_uncertainty_comparison_figure",
    "render_hypothesis_validation_tab",
]


def analyze_validation_results() -> dict:
    """Run the analysis script and parse results."""
    try:
        project_root = _PROJECT_ROOT
        script_path = _analysis_script_path()
        if not script_path.is_file():
            return {
                "success": False,
                "error": (
                    "Analysis script not found. Expected one of:\n"
                    f"  - {project_root / 'scripts' / 'legacy' / 'analyze_validation_results.py'}\n"
                    f"  - {project_root / 'analyze_validation_results.py'}"
                ),
            }

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            return {"success": True, "output": result.stdout}
        else:
            return {"success": False, "error": result.stderr}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def load_sweep_metrics(
    sweep_name: str,
    sources: tuple[str, ...] = ("pytorch_validation", "paper_keras"),
    dataset: str | None = None,
) -> pd.DataFrame:
    """Load unified metrics for a sweep (backward-compatible name)."""
    try:
        return load_unified_metrics(
            sweep_name,
            sources=sources,
            results_root=_PROJECT_ROOT / "results",
            dataset=dataset,
        )
    except Exception as e:
        st.error(f"Error loading {sweep_name} metrics: {e}")
        return pd.DataFrame()


def _metrics_csv_path(sweep_name: str) -> Path:
    return _PROJECT_ROOT / "results" / "validation" / f"{sweep_name}_sweep" / "metrics.csv"


def _discover_run_folders(*, limit: int = 30) -> list[Path]:
    """Recent experiment folders that have summary.json or results.pt."""
    candidates: list[Path] = []
    for sweep in ("dataset_size_sweep", "label_noise_sweep"):
        root = _PROJECT_ROOT / "results" / "validation" / sweep
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if (child / "summary.json").is_file() or (child / "results.pt").is_file():
                candidates.append(child)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[:limit]


def _render_inspect_run_folder() -> None:
    """Lightweight debugger: one folder, summary + per-sample table and plots."""
    with st.expander("Inspect one run folder (per-sample signals)", expanded=True):
        st.caption(
            "Aggregate plots use `metrics.csv`. Pick a run folder here to inspect "
            "`per_sample_signals.csv` (all signal columns by eval group). "
            "See `docs/debug/EVAL_ARTIFACTS.md`."
        )
        folders = _discover_run_folders()
        if not folders:
            st.info("No run folders found under `results/validation/*_sweep/`. Run a sweep first.")
            return

        labels = [str(p.relative_to(_PROJECT_ROOT)) for p in folders]
        choice = st.selectbox("Run folder", labels, key="hv_inspect_run")
        run_dir = _PROJECT_ROOT / choice
        artifacts = load_run_directory(run_dir)

        if not artifacts.has_data:
            st.warning("No `summary.json` or `results.pt` in this folder.")
            return

        st.caption(f"Loaded from: **{artifacts.source}**")
        col_a, col_b, col_c = st.columns(3)
        sizes = artifacts.eval_sizes
        with col_a:
            st.metric("Clean eval", sizes.get("clean", "—"))
        with col_b:
            st.metric("Aleatoric eval", sizes.get("aleatoric_like", "—"))
        with col_c:
            st.metric("Epistemic eval", sizes.get("epistemic_like", "—"))

        if artifacts.summary_path:
            with st.expander("summary.json", expanded=True):
                st.json(json.loads(artifacts.summary_path.read_text()))
        else:
            st.info("No `summary.json` (metrics read from `results.pt` only).")

        max_rows = st.slider(
            "Max rows to load",
            min_value=200,
            max_value=5000,
            value=1800,
            step=100,
            key="hv_per_sample_max_rows",
            help="Full eval packs are typically 1800 samples (600 per group).",
        )
        per_sample = load_per_sample_table(run_dir, max_rows=max_rows)
        if per_sample is not None:
            render_per_sample_signal_visualizations(per_sample)
        else:
            st.info("No `per_sample_signals.csv` in this folder.")


def _fallback_x_col(sweep_type: str) -> str:
    return "dataset_size" if sweep_type == "dataset_size" else "noise_percent"


def _pick_x_col(df: pd.DataFrame, sweep_type: str) -> str:
    """Resolve metrics x-axis column; safe when *df* is empty."""
    if df.empty:
        return _fallback_x_col(sweep_type)
    try:
        return resolve_x_col(df, sweep_type)
    except ValueError:
        return _fallback_x_col(sweep_type)


def _render_x_col_selector(df: pd.DataFrame, sweep_type: str) -> str:
    """Return x-axis column name; optional selectbox when label-noise has both variants."""
    x_col = _pick_x_col(df, sweep_type)
    if sweep_type != "label_noise" or df.empty:
        return x_col
    candidates = [c for c in ("noise_percent", "noise_rate") if c in df.columns]
    if len(candidates) < 2:
        return x_col
    default_x = x_col if x_col in candidates else candidates[0]
    return st.selectbox(
        "X-axis column",
        candidates,
        index=candidates.index(default_x),
        key="hv_v2_x_col",
        help="Some runs emit `noise_percent`, others `noise_rate`.",
    )


def _sweep_points_for_arch(
    df: pd.DataFrame,
    architecture: str,
    x_col: str,
) -> list:
    if df.empty or not architecture or x_col not in df.columns:
        return []
    arch_df = df[df["architecture"] == architecture]
    if arch_df.empty:
        return []
    return sorted(arch_df[x_col].dropna().unique().tolist())


def _render_missing_data_callout(sweep_name: str, sweep_label: str) -> None:
    csv_path = _metrics_csv_path(sweep_name)
    st.info(
        "\n".join(
            [
                f"**No {sweep_label} results found yet.**",
                "",
                "To generate them, run the experiments above, then come back here and refresh.",
                f"Expected file: `{csv_path}`",
            ]
        )
    )


def _render_aggregate_sweep_figure(
    df: pd.DataFrame,
    *,
    sweep_type: str,
    x_col: str,
    signal_metric,
    selected_architectures: list[str],
) -> None:
    if df.empty:
        _render_missing_data_callout(
            "dataset_size" if sweep_type == "dataset_size" else "label_noise",
            sweep_label="dataset size sweep" if sweep_type == "dataset_size" else "label noise sweep",
        )
        return

    if signal_metric.name == "auroc_only":
        _render_row3_signals_expander(df, sweep_type=sweep_type)

    fig = create_method_uncertainty_comparison_figure(
        df,
        x_col=x_col,
        sweep_type=sweep_type,
        architectures=selected_architectures if selected_architectures else None,
        signal_metric=signal_metric,
    )
    if fig is None:
        st.warning(
            "Could not build the comparison figure — check metrics CSV columns for this view."
        )
    else:
        st.plotly_chart(fig, use_container_width=True)

    if signal_metric.name == "auroc_only":
        with st.expander("AUROC summary stats", expanded=False):
            if sweep_type == "dataset_size":
                st.markdown("**Epistemic target signals**")
                for signal in EPISTEMIC_SIGNALS:
                    col_name = f"{signal}_epistemic_auroc"
                    if col_name in df.columns:
                        mean_auroc = df[col_name].mean()
                        std_auroc = df[col_name].std()
                        max_auroc = df[col_name].max()
                        status = "\u2705" if mean_auroc > 0.75 else "\u26a0\ufe0f"
                        st.metric(
                            f"{status} {signal}",
                            f"{mean_auroc:.3f}",
                            f"\u00b1{std_auroc:.3f} (max: {max_auroc:.3f})",
                        )
            else:
                st.markdown("**Aleatoric target signals**")
                for signal in ALEATORIC_SIGNALS:
                    col_name = f"{signal}_aleatoric_auroc"
                    if col_name in df.columns:
                        mean_auroc = df[col_name].mean()
                        std_auroc = df[col_name].std()
                        max_auroc = df[col_name].max()
                        status = "\u2705" if mean_auroc > 0.65 else "\u26a0\ufe0f"
                        st.metric(
                            f"{status} {signal}",
                            f"{mean_auroc:.3f}",
                            f"\u00b1{std_auroc:.3f} (max: {max_auroc:.3f})",
                        )


def _render_row3_signals_expander(df: pd.DataFrame, sweep_type: str) -> None:
    with st.expander("Row 3 selection (top signals)", expanded=False):
        st.caption(
            f"Row 3 selects up to 4 signals from: {', '.join(ROW3_CANDIDATE_SIGNALS)} "
            f"ranked by mean {sweep_to_auroc_type(sweep_type)} AUROC for this sweep."
        )
        if df.empty:
            st.warning("No metrics loaded yet for this sweep.")
            return
        ranked = get_row3_signals(df, sweep_type=sweep_type)
        if not ranked:
            st.warning("Could not rank Row 3 signals (missing AUROC columns).")
            return
        st.markdown("**Selected (in order):**")
        for signal, mean_auroc in ranked:
            label = SIGNAL_LABELS.get(signal, signal)
            st.write(f"- {label} (`{signal}`): mean AUROC = {mean_auroc:.3f}")



def render_hypothesis_validation_tab():
    """Render the hypothesis validation tab."""
    st.markdown("## 🔬 Hypothesis Validation Dashboard")
    st.markdown("""
    **Simple workflow:** (1) Run a sweep below → (2) Inspect one folder if something looks wrong →
    (3) Visualize from `metrics.csv`. One engine: `run_fast_uncertainty_classification.py`.
    """)
    st.markdown("---")
    st.markdown("### Run preset sweeps")
    render_preset_validation_sweeps(key_prefix="hv")
    # -------------------------
    # Analyze
    # -------------------------
    st.markdown("---")
    st.markdown("### 📊 Analyze")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        analyze_button = st.button(
            "Analyze results",
            key="analyze_results",
            help="Runs the hypothesis verification analysis script.",
            use_container_width=True,
            type="primary",
        )
    with col_b:
        refresh_button = st.button(
            "Refresh visualizations",
            key="refresh_plots",
            help="Reloads results from disk and redraws plots.",
            use_container_width=True,
        )

    if analyze_button:
        with st.spinner("Analyzing validation results..."):
            results = analyze_validation_results()
            if results["success"]:
                st.success("✅ Analysis complete!")
                with st.expander("Analysis output", expanded=False):
                    st.code(results["output"], language="text")
            else:
                st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")

    # -------------------------
    # Visualize
    # -------------------------
    st.markdown("---")
    st.markdown("### 📈 Visualize")

    # Load BOTH sources up front; sub-filters (dataset, methods, architectures)
    # are applied below in-memory.  Loading is cheap; the CSVs are small.
    epistemic_df_all = load_sweep_metrics(
        "dataset_size", sources=("pytorch_validation", "paper_keras")
    )
    aleatoric_df_all = load_sweep_metrics(
        "label_noise", sources=("pytorch_validation", "paper_keras")
    )

    # --- 1) Dataset selector (top-level) ---
    combined_for_discovery = pd.concat(
        [epistemic_df_all, aleatoric_df_all], ignore_index=True, sort=False
    )
    datasets_in_data = present_datasets(combined_for_discovery)
    if not datasets_in_data:
        # Nothing loaded yet; fall back to CIFAR-10 so the rest of the UI renders.
        datasets_in_data = ["cifar10"]

    default_dataset_idx = (
        datasets_in_data.index("cifar10") if "cifar10" in datasets_in_data else 0
    )
    selected_dataset = st.radio(
        "Dataset",
        options=datasets_in_data,
        index=default_dataset_idx,
        format_func=dataset_label,
        horizontal=True,
        key="hv_dataset",
        help=(
            "Filters every panel below to runs on this dataset. "
            "Both the PyTorch pipeline and the paper Keras code support multiple "
            "datasets, but only the datasets you've actually generated CSVs for "
            "appear here."
        ),
    )
    missing_datasets = [d for d in DATASETS if d not in datasets_in_data]
    if missing_datasets:
        missing_str = ", ".join(dataset_label(d) for d in missing_datasets)
        st.caption(
            f"_Not yet generated: {missing_str}._  "
            "The paper Keras code supports Fashion-MNIST natively "
            "(`disentanglement_paper/data/fashion_mnist.py`); your PyTorch pipeline "
            "currently only supports CIFAR-10 / CIFAR-10N."
        )

    # Apply the dataset filter to both DataFrames.
    def _filter_by_dataset(df: pd.DataFrame, ds: str) -> pd.DataFrame:
        if "dataset" in df.columns and not df.empty:
            return df[df["dataset"] == ds].copy()
        return df

    epistemic_df = _filter_by_dataset(epistemic_df_all, selected_dataset)
    aleatoric_df = _filter_by_dataset(aleatoric_df_all, selected_dataset)

    sweep_choice = st.radio(
        "Sweep",
        ["Dataset size", "Label noise"],
        index=0 if not epistemic_df.empty else 1,
        horizontal=True,
        key="hv_sweep_choice",
    )

    sweep_type = "dataset_size" if sweep_choice == "Dataset size" else "label_noise"
    df = epistemic_df if sweep_type == "dataset_size" else aleatoric_df

    # Always initialize before any widget block (avoids UnboundLocalError on rerun).
    x_col = _fallback_x_col(sweep_type)
    selected_sources: list[str] = []
    selected_architectures: list[str] = []
    architecture = ""
    sweep_point: float | int = 0
    selected_signals: list[str] = list(FAST_PILOT_SIGNAL_NAMES)
    x_values: list = []

    x_col = _render_x_col_selector(df, sweep_type)

    if not df.empty:
        col_methods, col_arch, col_point, col_sig = st.columns([1, 1, 1, 1])

        sources_present = present_sources(df)
        with col_methods:
            pytorch_default = (
                ["pytorch_validation"]
                if "pytorch_validation" in sources_present
                else sources_present
            )
            selected_sources = st.multiselect(
                "Methods",
                options=sources_present,
                default=pytorch_default,
                format_func=_source_label,
                key="hv_v2_sources",
                help="Per-sample decision tables need **Your PyTorch attribution** runs.",
            )
        if not selected_sources:
            selected_sources = sources_present

        if "source" in df.columns:
            df = df[df["source"].isin(selected_sources)].copy()

        x_col = _pick_x_col(df, sweep_type) if not df.empty else x_col

        with col_arch:
            arch_options = present_architectures(df)
            if not arch_options:
                st.warning("No architectures after method filter.")
                architecture = ""
            else:
                default_arch = (
                    "DINOv2 + MLP"
                    if "DINOv2 + MLP" in arch_options
                    else arch_options[0]
                )
                architecture = st.selectbox(
                    "Architecture",
                    arch_options,
                    index=arch_options.index(default_arch),
                    key="hv_v2_architecture",
                    help="Row 1 + each signal row use this architecture only.",
                )
            selected_architectures = [architecture] if architecture else []

        x_values = _sweep_points_for_arch(df, architecture, x_col)

        with col_point:
            if x_values:
                sweep_point = st.selectbox(
                    "Sweep point (per-sample tables)",
                    x_values,
                    key="hv_v2_sweep_point",
                )
            else:
                st.selectbox("Sweep point", ["—"], key="hv_v2_sweep_point_empty", disabled=True)
                sweep_point = 0

    # --- 4) Disentanglement status caption (after method/arch filtering) ---
    if not df.empty:
        dis_present = present_disentanglements(df)
        if dis_present:
            dis_labels = " · ".join(disentanglement_label(d) for d in dis_present)
            st.caption(f"Rows rendered for: **{dis_labels}**")
            paper_dis = [d for d in dis_present if d in ("gaussian_logits", "information_theoretic")]
            if not paper_dis and "paper_keras" not in selected_sources:
                st.info(
                    "Only **PyTorch Attribution** rows are showing. To see the "
                    "paper's Gaussian-Logits / Information-Theoretic rows, tick "
                    "**Paper Keras (reference)** in the *Methods* selector — or "
                    "run `scripts/run_paper_benchmarks.py` if no Keras CSVs exist yet."
                )

    st.caption(
        "**Row 1** — architecture uncertainty across the sweep. **Signal rows** — col 1: "
        "that signal's epistemic/aleatoric means; cols 2–3: samples misclassified in "
        "aleatoric vs epistemic eval pools (`per_sample_signals.csv`)."
    )

    if df.empty:
        _render_missing_data_callout(
            "dataset_size" if sweep_type == "dataset_size" else "label_noise",
            sweep_label="dataset size sweep" if sweep_type == "dataset_size" else "label noise sweep",
        )
    elif not architecture:
        st.warning("Select an architecture with metrics in this sweep.")
    elif not x_values:
        st.warning(
            f"No sweep points for **{architecture}** on `{x_col}`. "
            "Run experiments for this architecture or change Methods filter."
        )
    else:
        render_visualize_main(
            df,
            project_root=_PROJECT_ROOT,
            sweep_type=sweep_type,
            x_col=x_col,
            architecture=architecture,
            sweep_point=sweep_point,
            selected_signals=selected_signals or list(FAST_PILOT_SIGNAL_NAMES),
        )

    with st.expander("Compare multiple architectures (sweep uncertainty grid)", expanded=False):
        compare_archs = st.multiselect(
            "Architectures for grid",
            present_architectures(df) if not df.empty else [],
            default=selected_architectures if selected_architectures else None,
            key="hv_v2_compare_archs",
        )
        if compare_archs:
            _render_aggregate_sweep_figure(
                df,
                sweep_type=sweep_type,
                x_col=x_col,
                signal_metric=UNCERTAINTY_DECOMPOSITION,
                selected_architectures=compare_archs,
            )

    with st.expander("AUROC grid (aggregate discrimination)", expanded=False):
        auroc_archs = st.multiselect(
            "Architectures for AUROC grid",
            present_architectures(df) if not df.empty else [],
            default=selected_architectures if selected_architectures else None,
            key="hv_v2_auroc_archs",
        )
        if auroc_archs:
            _render_aggregate_sweep_figure(
                df,
                sweep_type=sweep_type,
                x_col=x_col,
                signal_metric=AUROC_ONLY,
                selected_architectures=auroc_archs,
            )

    _render_inspect_run_folder()

    # Help section
    st.markdown("---")
    with st.expander("ℹ️ How to Use This Dashboard"):
        st.markdown("""
        ### Step-by-Step Guide
        
        1. **Choose Experiment Type:**
           - **Epistemic Validation**: Tests if signals detect model uncertainty
           - **Aleatoric Validation**: Tests if signals detect data noise
        
        2. **Select Mode:**
           - **Quick**: 3-5 experiments, ~10 minutes
           - **Full**: 10+ experiments, ~1 hour
        
        3. **Run Experiments:**
           - Click the appropriate "Run" button
           - Wait for experiments to complete
           - Check output for any errors
        
        4. **Analyze Results:**
           - Click "Analyze Results" to run hypothesis verification
           - Review the text output for pass/fail status
           - Click "Refresh Plots" to see visualizations
        
        5. **Interpret Results:**
           - **Green lines**: Target signals (should perform well)
           - **Blue lines**: Other signals (for comparison)
           - **Red dashed line**: Target AUROC threshold
           - **Above threshold + increasing trend = Success! ✅**
        
        ### Expected Behavior
        
        **Epistemic Signals** (`inverse_mass`, `dominance`):
        - Should show AUROC > 0.75 in dataset size sweep
        - Should increase as dataset size increases
        - Should remain stable in noise sweep
        
        **Aleatoric Signals** (`inverse_coherence`):
        - Should show AUROC > 0.65 in noise sweep
        - Should increase as noise increases
        - Should remain stable in dataset size sweep
        """)

# Made with Bob