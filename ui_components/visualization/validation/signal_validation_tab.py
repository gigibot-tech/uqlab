"""
Signal validation dashboard for the progressive Streamlit app.

Runs preset dataset-size / label-noise sweeps, visualizes metric and AUROC plots,
and surfaces four-region monotonicity / orthogonality checks (Aspect 7).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from uqlab.notebook_support.metric_specs import AUROC_ONLY, UNCERTAINTY_DECOMPOSITION
from uqlab.notebook_support.method_comparison_plotly import (
    create_method_uncertainty_comparison_figure,
)
from uqlab.notebook_support.signals import (
    ALEATORIC_SIGNALS,
    EPISTEMIC_SIGNALS,
    ROW3_CANDIDATE_SIGNALS,
    SIGNAL_LABELS,
    SIGNAL_NAMES,
    disentanglement_label,
    get_row3_signals,
    present_architectures,
    present_datasets,
    present_disentanglements,
    present_sources,
    resolve_x_col,
    sweep_to_auroc_type,
)
from uqlab.results_io import DATASETS, dataset_label, load_unified_metrics
from uqlab.run_artifacts import FAST_PILOT_SIGNAL_NAMES, load_per_sample_table, load_run_directory
from uqlab.ui_components.workflow.validation_runner import (
    render_preset_validation_sweeps,
)


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").is_file() and (p / "scripts").is_dir():
            return p
    return here.parents[4]


_PROJECT_ROOT = _resolve_project_root()
_SRC = _PROJECT_ROOT / "src"
for _p in (_SRC, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_SOURCE_LABELS = {
    "pytorch_validation": "Your PyTorch attribution",
    "paper_keras": "Paper Keras (reference)",
}


def _source_label(key: str) -> str:
    return _SOURCE_LABELS.get(key, key.replace("_", " ").title())


def load_sweep_metrics(
    sweep_name: str,
    sources: tuple[str, ...] = ("pytorch_validation", "paper_keras"),
    dataset: str | None = None,
) -> pd.DataFrame:
    try:
        return load_unified_metrics(
            sweep_name,
            sources=sources,
            results_root=_PROJECT_ROOT / "results",
            dataset=dataset,
        )
    except Exception as exc:
        st.error(f"Error loading {sweep_name} metrics: {exc}")
        return pd.DataFrame()


def _metrics_csv_path(sweep_name: str) -> Path:
    return _PROJECT_ROOT / "results" / "validation" / f"{sweep_name}_sweep" / "metrics.csv"


def _discover_run_folders(*, limit: int = 30) -> list[Path]:
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
    with st.expander("Inspect one run folder (per-sample signals)", expanded=False):
        st.caption(
            "Pick a run folder to inspect `per_sample_signals.csv` (all signal columns by eval group)."
        )
        folders = _discover_run_folders()
        if not folders:
            st.info("No run folders found under `results/validation/*_sweep/`. Run a sweep first.")
            return

        labels = [str(p.relative_to(_PROJECT_ROOT)) for p in folders]
        choice = st.selectbox("Run folder", labels, key="sv_inspect_run")
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
            with st.expander("summary.json", expanded=False):
                st.json(json.loads(artifacts.summary_path.read_text()))

        max_rows = st.slider(
            "Max rows to load",
            min_value=200,
            max_value=5000,
            value=1800,
            step=100,
            key="sv_per_sample_max_rows",
        )
        per_sample = load_per_sample_table(run_dir, max_rows=max_rows)
        if per_sample is not None:
            from uqlab.ui_components.visualization.signals.per_sample_signals_viz import (
                render_per_sample_signal_visualizations,
            )

            render_per_sample_signal_visualizations(per_sample)
        else:
            st.info("No `per_sample_signals.csv` in this folder.")


def _fallback_x_col(sweep_type: str) -> str:
    return "dataset_size" if sweep_type == "dataset_size" else "noise_percent"


def _pick_x_col(df: pd.DataFrame, sweep_type: str) -> str:
    if df.empty:
        return _fallback_x_col(sweep_type)
    try:
        return resolve_x_col(df, sweep_type)
    except ValueError:
        return _fallback_x_col(sweep_type)


def _render_x_col_selector(df: pd.DataFrame, sweep_type: str) -> str:
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
        key="sv_x_col",
    )


def _render_missing_data_callout(sweep_name: str, sweep_label: str) -> None:
    csv_path = _metrics_csv_path(sweep_name)
    st.info(
        f"**No {sweep_label} results found yet.**\n\n"
        f"Run the sweeps above, then refresh. Expected: `{csv_path}`"
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
        st.warning("Could not build the comparison figure — check metrics CSV columns.")
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
                        status = "✅" if mean_auroc > 0.75 else "⚠️"
                        st.metric(
                            f"{status} {signal}",
                            f"{mean_auroc:.3f}",
                            f"±{std_auroc:.3f} (max: {max_auroc:.3f})",
                        )
            else:
                st.markdown("**Aleatoric target signals**")
                for signal in ALEATORIC_SIGNALS:
                    col_name = f"{signal}_aleatoric_auroc"
                    if col_name in df.columns:
                        mean_auroc = df[col_name].mean()
                        std_auroc = df[col_name].std()
                        max_auroc = df[col_name].max()
                        status = "✅" if mean_auroc > 0.65 else "⚠️"
                        st.metric(
                            f"{status} {signal}",
                            f"{mean_auroc:.3f}",
                            f"±{std_auroc:.3f} (max: {max_auroc:.3f})",
                        )


def _render_row3_signals_expander(df: pd.DataFrame, sweep_type: str) -> None:
    with st.expander("Row 3 selection (top signals)", expanded=False):
        if df.empty:
            st.warning("No metrics loaded yet for this sweep.")
            return
        ranked = get_row3_signals(df, sweep_type=sweep_type)
        if not ranked:
            st.warning("Could not rank Row 3 signals (missing AUROC columns).")
            return
        for signal, mean_auroc in ranked:
            label = SIGNAL_LABELS.get(signal, signal)
            st.write(f"- {label} (`{signal}`): mean AUROC = {mean_auroc:.3f}")


def _render_preset_sweep_visualize() -> None:
    st.markdown("### Visualize preset sweeps")

    epistemic_df_all = load_sweep_metrics(
        "dataset_size", sources=("pytorch_validation", "paper_keras")
    )
    aleatoric_df_all = load_sweep_metrics(
        "label_noise", sources=("pytorch_validation", "paper_keras")
    )

    combined_for_discovery = pd.concat(
        [epistemic_df_all, aleatoric_df_all], ignore_index=True, sort=False
    )
    datasets_in_data = present_datasets(combined_for_discovery)
    if not datasets_in_data:
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
        key="sv_dataset",
    )
    missing_datasets = [d for d in DATASETS if d not in datasets_in_data]
    if missing_datasets:
        missing_str = ", ".join(dataset_label(d) for d in missing_datasets)
        st.caption(f"_Not yet generated: {missing_str}._")

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
        key="sv_sweep_choice",
    )

    sweep_type = "dataset_size" if sweep_choice == "Dataset size" else "label_noise"
    df = epistemic_df if sweep_type == "dataset_size" else aleatoric_df

    x_col = _render_x_col_selector(df, sweep_type)
    selected_architectures: list[str] = []
    architecture = ""
    sweep_point: float | int = 0
    x_values: list = []

    if not df.empty:
        col_methods, col_arch, col_point = st.columns([1, 1, 1])
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
                key="sv_sources",
            )
        if not selected_sources:
            selected_sources = sources_present
        if "source" in df.columns:
            df = df[df["source"].isin(selected_sources)].copy()
        x_col = _pick_x_col(df, sweep_type) if not df.empty else x_col

        with col_arch:
            arch_options = present_architectures(df)
            if arch_options:
                default_arch = (
                    "DINOv2 + MLP" if "DINOv2 + MLP" in arch_options else arch_options[0]
                )
                architecture = st.selectbox(
                    "Architecture",
                    arch_options,
                    index=arch_options.index(default_arch),
                    key="sv_architecture",
                )
                selected_architectures = [architecture]
        x_values = (
            sorted(df[df["architecture"] == architecture][x_col].dropna().unique().tolist())
            if architecture and x_col in df.columns
            else []
        )
        with col_point:
            if x_values:
                sweep_point = st.selectbox(
                    "Sweep point (per-sample tables)",
                    x_values,
                    key="sv_sweep_point",
                )

    if not df.empty:
        dis_present = present_disentanglements(df)
        if dis_present:
            dis_labels = " · ".join(disentanglement_label(d) for d in dis_present)
            st.caption(f"Rows rendered for: **{dis_labels}**")

    if df.empty:
        _render_missing_data_callout(
            "dataset_size" if sweep_type == "dataset_size" else "label_noise",
            sweep_label="dataset size sweep" if sweep_type == "dataset_size" else "label noise sweep",
        )
    elif not architecture:
        st.warning("Select an architecture with metrics in this sweep.")
    elif not x_values:
        st.warning(f"No sweep points for **{architecture}** on `{x_col}`.")
    else:
        from uqlab.ui_components.visualization.signals.signal_diagnostic_viz import (
            render_visualize_main,
        )

        render_visualize_main(
            df,
            project_root=_PROJECT_ROOT,
            sweep_type=sweep_type,
            x_col=x_col,
            architecture=architecture,
            sweep_point=sweep_point,
            selected_signals=list(FAST_PILOT_SIGNAL_NAMES),
        )

    with st.expander("Compare multiple architectures (sweep uncertainty grid)", expanded=False):
        compare_archs = st.multiselect(
            "Architectures for grid",
            present_architectures(df) if not df.empty else [],
            default=selected_architectures if selected_architectures else None,
            key="sv_compare_archs",
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
            key="sv_auroc_archs",
        )
        if auroc_archs:
            _render_aggregate_sweep_figure(
                df,
                sweep_type=sweep_type,
                x_col=x_col,
                signal_metric=AUROC_ONLY,
                selected_architectures=auroc_archs,
            )


def _coerce_four_region_payload(
    payload: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize an uploaded JSON into ``(noise_rows, sparsity_rows)``.

    Accepts the canonical ``{"noise_rows", "sparsity_rows"}`` dict (incl. exported
    reports), a bare list of row dicts, or a single row dict — splitting rows by
    the presence of ``noise_pct`` / ``sparse_train_pct``.
    """
    def _split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        noise = [r for r in rows if isinstance(r, dict) and "noise_pct" in r]
        sparse = [r for r in rows if isinstance(r, dict) and "sparse_train_pct" in r]
        return noise, sparse

    if isinstance(payload, dict):
        noise_rows = payload.get("noise_rows")
        sparsity_rows = payload.get("sparsity_rows")
        if noise_rows is not None or sparsity_rows is not None:
            return list(noise_rows or []), list(sparsity_rows or [])
        # Single row dict — route by its sweep key.
        return _split([payload])
    if isinstance(payload, list):
        return _split([r for r in payload if isinstance(r, dict)])
    return [], []


_REGION_ORDER = ["clean", "aleatoric_like", "epistemic_like", "ood_like"]


def _render_per_region_signal_means(run_dir: Path, *, key_prefix: str) -> None:
    """Simple metric plots: mean of each signal per region from per_sample_signals.csv."""
    csv_path = run_dir / "per_sample_signals.csv"
    if not csv_path.is_file():
        return
    try:
        df = pd.read_csv(csv_path)
    except (OSError, ValueError) as exc:
        st.caption(f"Could not read `{csv_path.name}`: {exc}")
        return
    if "group" not in df.columns:
        return

    drop = {"group", "dataset_index", "clean_label", "noisy_label", "is_noisy"}
    signal_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    if not signal_cols:
        return

    means = df.groupby("group")[signal_cols].mean()
    order = [g for g in _REGION_ORDER if g in means.index] + [
        g for g in means.index if g not in _REGION_ORDER
    ]
    means = means.reindex(order)

    st.markdown("**Mean signal per region** (from `per_sample_signals.csv`)")
    st.dataframe(means.round(4), use_container_width=True)

    long_df = means.reset_index().melt(id_vars="group", var_name="signal", value_name="mean")
    fig = px.bar(
        long_df,
        x="group",
        y="mean",
        color="group",
        facet_col="signal",
        facet_col_wrap=4,
        category_orders={"group": order},
        title="Mean uncertainty by region (per signal)",
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(showticklabels=False, title=None)
    fig.update_layout(height=420, showlegend=True)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_region_means")


def _render_four_region_run_report(
    summary: dict[str, Any],
    *,
    key_prefix: str,
    source_label: str = "uploaded run",
    run_dir: Path | None = None,
) -> None:
    """Per-region AUROC report for a single four-region run summary.

    A four-region run separates noisy (aleatoric), sparse (epistemic), and OOD
    regions from clean in one shot — so the validation here is region separability
    per signal, not a multi-point sweep.
    """
    from uqlab.evaluation.four_region_validation import four_region_run_auroc_rows

    rows = four_region_run_auroc_rows(summary)
    if not rows:
        st.warning(f"`{source_label}` has no per-signal region AUROCs to report.")
        if run_dir is not None:
            _render_per_region_signal_means(run_dir, key_prefix=key_prefix)
        return

    st.caption(
        f"Single four-region run ({source_label}). Region separability per signal: "
        "good **aleatoric** signals score high on noisy↔clean & low on sparse↔clean; "
        "good **epistemic** signals are the reverse."
    )
    df = pd.DataFrame(rows)
    sort_col = "specialization" if "specialization" in df.columns else "signal"
    st.dataframe(
        df.sort_values(sort_col, ascending=False) if sort_col != "signal" else df,
        use_container_width=True,
        hide_index=True,
    )

    alea_col, epi_col = "aleatoric (noisy↔clean)", "epistemic (sparse↔clean)"
    if alea_col in df.columns and epi_col in df.columns:
        plot_df = df.dropna(subset=[alea_col, epi_col])
        if not plot_df.empty:
            fig = px.scatter(
                plot_df,
                x=alea_col,
                y=epi_col,
                text="signal",
                title="Region separability — aleatoric (x) vs epistemic (y)",
            )
            fig.update_traces(textposition="top center")
            fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
            fig.add_vline(x=0.5, line_dash="dot", line_color="gray")
            fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_run_scatter")

    if run_dir is not None:
        _render_per_region_signal_means(run_dir, key_prefix=key_prefix)

    with st.expander("Raw run summary (JSON)", expanded=False):
        st.json(summary)


def render_four_region_validation_panel(
    *,
    key_prefix: str = "sv_fr",
    project_root: Path | None = None,
) -> None:
    """Aspect 7: monotonic + orthogonal correlation report."""
    from uqlab.evaluation.four_region_validation import (
        DEFAULT_ALEATORIC_METRICS,
        DEFAULT_EPISTEMIC_METRICS,
        build_correlation_report,
        discover_four_region_run_dirs,
        load_four_region_rows_from_sweep_csvs,
        load_four_region_sweep_rows_from_disk,
        mock_sweep_metric_rows,
        noise_sweep_regions,
        report_to_json,
        run_four_region_sweep_inprocess,
        sparsity_sweep_regions,
        sweep_csv_architectures,
    )

    root = project_root or _PROJECT_ROOT
    results_root = root / "results"
    validation_dir = root / "results" / "validation"
    four_region_dir = validation_dir / "four_region"
    noise_csv = validation_dir / "label_noise_sweep" / "metrics.csv"
    sparsity_csv = validation_dir / "dataset_size_sweep" / "metrics.csv"

    st.markdown("### Four-region signal validation (Aspect 7)")
    st.caption(
        "Vary noisy-region label flip % and sparse-region train fraction; "
        "check that aleatoric metrics track noise, epistemic metrics track sparsity, "
        "and the sweeps are orthogonal."
    )

    with st.expander("Sweep axes (dedicated four-region runs)", expanded=False):
        st.markdown("**Noise sweep** (noisy region `label_flip_pct`):")
        for pct, _ in noise_sweep_regions():
            st.write(f"- {pct}%")
        st.markdown("**Sparsity sweep** (sparse region `train_fraction`):")
        for pct, regions in sparsity_sweep_regions():
            frac = regions["sparse"]["train_fraction"]
            st.write(f"- {pct}% ({frac})")

    aleatoric_metrics: list[str] = list(DEFAULT_ALEATORIC_METRICS)
    epistemic_metrics: list[str] = list(DEFAULT_EPISTEMIC_METRICS)

    source = st.radio(
        "Report source",
        ["Load from results/", "Upload JSON", "Synthetic example", "Run quick sweep (GPU)"],
        horizontal=True,
        key=f"{key_prefix}_source",
    )

    noise_rows: list[dict[str, Any]] = []
    sparsity_rows: list[dict[str, Any]] = []

    if source == "Synthetic example":
        st.caption(
            "Illustrative **fake** data showing an ideal PASS layout — not your runs. "
            "Use *Load from results/* for real metrics."
        )
        noise_rows, sparsity_rows = mock_sweep_metric_rows()
    elif source == "Load from results/":
        # 0) Single four-region runs (per-region AUROC report) — what most runs produce.
        run_summaries = discover_four_region_run_dirs(results_root)
        if run_summaries:
            labels = [str(p.parent.relative_to(results_root)) for p in run_summaries]
            choices = ["— sweep (monotonic/orthogonal) —", *labels]
            chosen = st.selectbox(
                "Four-region run (single-run region report)",
                choices,
                key=f"{key_prefix}_run_pick",
                help="Single four-region runs report region separability per signal. "
                "Pick the first option for a multi-point sweep correlation instead.",
            )
            if chosen != choices[0]:
                summary_path = run_summaries[labels.index(chosen)]
                try:
                    summary = json.loads(summary_path.read_text())
                except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                    st.error(f"Could not read `{summary_path}`: {exc}")
                    return
                _render_four_region_run_report(
                    summary,
                    key_prefix=f"{key_prefix}_disk",
                    source_label=chosen,
                    run_dir=summary_path.parent,
                )
                return

        # 1) Dedicated four-region partition sweep runs, if any exist.
        noise_rows, sparsity_rows = load_four_region_sweep_rows_from_disk(four_region_dir)
        if noise_rows or sparsity_rows:
            st.caption("Source: dedicated four-region runs under `results/validation/four_region/`.")
        else:
            # 2) Fall back to existing global sweeps as a proxy.
            archs = sweep_csv_architectures(noise_csv, sparsity_csv)
            have_csv = noise_csv.is_file() or sparsity_csv.is_file()
            if not have_csv:
                st.info(
                    "No four-region runs and no global sweep aggregates found under "
                    "`results/validation/` (`label_noise_sweep/metrics.csv`, "
                    "`dataset_size_sweep/metrics.csv`). Run a sweep or use *Synthetic example*."
                )
            else:
                arch = None
                if archs:
                    arch = st.selectbox(
                        "Architecture",
                        archs,
                        key=f"{key_prefix}_csv_arch",
                        help="Global sweeps store one curve per architecture.",
                    )
                noise_rows, sparsity_rows, aleatoric_metrics, epistemic_metrics = (
                    load_four_region_rows_from_sweep_csvs(
                        noise_csv=noise_csv if noise_csv.is_file() else None,
                        sparsity_csv=sparsity_csv if sparsity_csv.is_file() else None,
                        architecture=arch,
                    )
                )
                st.caption(
                    "Source: **global** label-noise + dataset-size sweeps "
                    "(`results/validation/{label_noise,dataset_size}_sweep`), used as a "
                    "proxy — these are not region-partitioned runs."
                )
                if not noise_rows and not sparsity_rows:
                    st.warning(
                        "Sweep CSVs found but no usable rows for this architecture "
                        "(need `noise_percent` / `under_train_per_class` + uncertainty means)."
                    )
    elif source == "Upload JSON":
        from uqlab.evaluation.four_region_validation import is_four_region_run_summary

        uploaded = st.file_uploader(
            "four-region run `summary.json`, or sweep JSON (`noise_rows` + `sparsity_rows`)",
            type=["json"],
            key=f"{key_prefix}_json",
        )
        st.caption(
            "Accepts a single four-region run `summary.json` (per-region AUROC report), "
            "a sweep report `{\"noise_rows\": […], \"sparsity_rows\": […]}`, a bare list "
            "of rows, or a single sweep-row dict."
        )
        if uploaded is not None:
            try:
                payload = json.loads(uploaded.getvalue().decode())
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                st.error(f"Could not parse `{uploaded.name}` as JSON: {exc}")
                payload = None
            if payload is not None and is_four_region_run_summary(payload):
                _render_four_region_run_report(
                    payload, key_prefix=key_prefix, source_label=uploaded.name
                )
                return
            if payload is not None:
                noise_rows, sparsity_rows = _coerce_four_region_payload(payload)
                if not noise_rows and not sparsity_rows:
                    st.warning(
                        f"`{uploaded.name}` is not a four-region run summary and has no "
                        "sweep rows. Need `one_vs_rest_auroc`/`auroc_rows` (single run) or "
                        "`noise_rows`/`sparsity_rows` (sweep)."
                    )
    else:
        from uqlab.evaluation.four_region_validation import FOUR_REGION_ARCHITECTURES

        col_mode, col_arch = st.columns(2)
        mode = col_mode.selectbox("Run mode", ["quick", "full"], key=f"{key_prefix}_mode")
        architecture = col_arch.selectbox(
            "Architecture",
            list(FOUR_REGION_ARCHITECTURES),
            key=f"{key_prefix}_run_arch",
            help="cnn_mcdropout/resnet18_mcdropout train end-to-end; dinov2_mlp uses DINOv2 features.",
        )
        col_n, col_s = st.columns(2)
        run_noise = col_n.button("Run noise sweep", key=f"{key_prefix}_run_noise", type="primary")
        run_sparse = col_s.button("Run sparsity sweep", key=f"{key_prefix}_run_sparse")
        if run_noise or run_sparse:
            sweep_kind = "noise" if run_noise else "sparsity"
            with st.status(f"Running four-region {sweep_kind} sweep…", expanded=True) as status:
                log_lines: list[str] = []

                def _on_line(line: str) -> None:
                    log_lines.append(line)
                    if len(log_lines) > 200:
                        log_lines.pop(0)

                ok, output = run_four_region_sweep_inprocess(
                    sweep_kind,
                    mode,
                    output_base=four_region_dir,
                    architecture=architecture,
                    on_line=_on_line,
                )
                st.code("\n".join(log_lines[-80:]) or output[-4000:], language="text")
                status.update(
                    label=f"{'Done' if ok else 'Failed'}: {sweep_kind} sweep",
                    state="complete" if ok else "error",
                )
        noise_rows, sparsity_rows = load_four_region_sweep_rows_from_disk(
            four_region_dir
        )

    if not noise_rows and not sparsity_rows:
        return

    report = build_correlation_report(
        noise_rows,
        sparsity_rows,
        aleatoric_metrics=aleatoric_metrics,
        epistemic_metrics=epistemic_metrics,
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Monotonic checks", "PASS" if report.monotonic_passed else "FAIL")
    with col_b:
        st.metric("Orthogonal checks", "PASS" if report.orthogonal_passed else "FAIL")
    with col_c:
        st.metric("Overall", "PASS" if report.monotonic_passed and report.orthogonal_passed else "FAIL")

    corr_df = pd.DataFrame(
        [
            {
                "sweep": row.sweep_kind,
                "metric": row.metric,
                "check": row.check,
                "spearman_r": row.spearman_r,
                "p_value": row.p_value,
                "passed": row.passed,
            }
            for row in report.correlations
        ]
    )
    st.dataframe(corr_df, use_container_width=True, hide_index=True)

    with st.expander("Raw JSON report", expanded=False):
        st.code(report_to_json(report), language="json")

    def _prioritize(metrics: list[str], preferred: list[str]) -> list[str]:
        pref = [m for m in preferred if m in metrics]
        return pref + [m for m in metrics if m not in pref]

    plot_metrics = (
        _prioritize(aleatoric_metrics, ["aleatoric_uncertainty"])[:2]
        + _prioritize(epistemic_metrics, ["epistemic_uncertainty"])[:2]
    )
    if noise_rows:
        noise_df = pd.DataFrame(noise_rows)
        if "noise_pct" in noise_df.columns:
            for metric in plot_metrics:
                if metric not in noise_df.columns:
                    continue
                fig = px.line(
                    noise_df,
                    x="noise_pct",
                    y=metric,
                    markers=True,
                    title=f"{metric} vs noise %",
                )
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_noise_{metric}")

    if sparsity_rows:
        sparse_df = pd.DataFrame(sparsity_rows)
        if "sparse_train_pct" in sparse_df.columns:
            for metric in plot_metrics:
                if metric not in sparse_df.columns:
                    continue
                fig = px.line(
                    sparse_df,
                    x="sparse_train_pct",
                    y=metric,
                    markers=True,
                    title=f"{metric} vs sparsity %",
                )
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_sparse_{metric}")


def _render_four_region_correlation_panel() -> None:
    render_four_region_validation_panel(key_prefix="sv_fr", project_root=_PROJECT_ROOT)


def render_signal_validation_tab() -> None:
    """Main entry for the Signal validation mode in the progressive app."""
    st.markdown("## Signal validation")
    st.markdown(
        "Validate uncertainty signals with preset sweeps (Fig 3/4 style) and "
        "four-region monotonicity / orthogonality checks."
    )
    st.markdown("---")
    st.markdown("### Run preset sweeps")
    render_preset_validation_sweeps(key_prefix="sv", show_viz=True)
    st.markdown("---")
    _render_preset_sweep_visualize()
    st.markdown("---")
    _render_four_region_correlation_panel()
    st.markdown("---")
    _render_inspect_run_folder()

    with st.expander("How to interpret", expanded=False):
        st.markdown(
            """
**Preset sweeps**
- Dataset size (under-train): epistemic signals should rise with more training data.
- Label noise: aleatoric signals should rise with noise %.

**Four-region (Aspect 7)**
- Noise sweep on the noisy region: aleatoric metrics should increase monotonically.
- Sparsity sweep on the sparse region: epistemic metrics should increase as data decreases.
- Orthogonal: noise sweep should not move epistemic metrics; sparsity sweep should not move aleatoric metrics.
            """
        )


__all__ = ["render_signal_validation_tab", "render_four_region_validation_panel", "SIGNAL_NAMES"]
