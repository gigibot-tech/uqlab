"""
Per-signal diagnostic panel: one architecture, sweep mini-plot, failure tables.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

from uqlab.notebook_support.metric_specs import (
    ACCURACY_COLOR,
    ALEATORIC_COLOR,
    ARCHITECTURE_ROW,
    EPISTEMIC_COLOR,
    UNCERTAINTY_DECOMPOSITION,
    resolve_columns,
)
from uqlab.notebook_support.signals import SIGNAL_LABELS, get_row3_signals, sweep_to_auroc_type
from uqlab.run_artifacts import FAST_PILOT_SIGNAL_NAMES, load_per_sample_table, load_run_directory
from uqlab_orchestrator.config.validation_config import ARCHITECTURES, LEGACY_ARCHITECTURE_LABELS

GROUP_STR_TO_INT = {
    "clean": 0,
    "aleatoric_like": 1,
    "epistemic_like": 2,
}
GROUP_INT_TO_STR = {v: k for k, v in GROUP_STR_TO_INT.items()}

POOL_TABLE_COLUMNS = (
    "dataset_index",
    "clean_label",
    "noisy_label",
    "is_noisy",
    "true_group",
    "predicted_group",
    "correct",
    "is_test_split",
)

CLASSIFICATION_METHOD_MD = """
**3-way linear classifier (cols 2–3)** — one scalar signal per run, not the plot uncertainty:

1. Map eval pack `group` → class: `clean`=0, `aleatoric_like`=1, `epistemic_like`=2.
2. **Standardize** the signal on a balanced 50/50 train split (same seed as `summary.json`).
3. Fit `Linear(1 → 3)` with Adam (300 steps) on the train split only.
4. Predict **every** sample; `correct` = predicted pack equals ground-truth `group`.

**Col 1 curves** — from `metrics.csv` at each sweep point: **mean** epistemic / aleatoric
uncertainty aggregated over the full eval set (MC dropout at inference), plus accuracy when present.
"""


def pool_classification_stats(df: pd.DataFrame, true_group: str) -> dict[str, int | float | None]:
    """Counts for one eval pool after :func:`enrich_per_sample_with_predictions`."""
    if df.empty or "true_group" not in df.columns:
        return {
            "n_pool": 0,
            "n_correct": 0,
            "n_misclassified": 0,
            "n_train": 0,
            "n_test": 0,
            "accuracy": None,
        }
    pool = df.loc[df["true_group"] == true_group]
    n_pool = len(pool)
    if n_pool == 0 or "correct" not in pool.columns:
        return {
            "n_pool": n_pool,
            "n_correct": 0,
            "n_misclassified": 0,
            "n_train": 0,
            "n_test": 0,
            "accuracy": None,
        }
    n_correct = int(pool["correct"].sum())
    n_misclassified = n_pool - n_correct
    n_train = int((~pool["is_test_split"]).sum()) if "is_test_split" in pool.columns else 0
    n_test = int(pool["is_test_split"].sum()) if "is_test_split" in pool.columns else 0
    return {
        "n_pool": n_pool,
        "n_correct": n_correct,
        "n_misclassified": n_misclassified,
        "n_train": n_train,
        "n_test": n_test,
        "accuracy": n_correct / n_pool,
    }


def filter_eval_pool(
    df: pd.DataFrame,
    true_group: str,
    *,
    only_misclassified: bool,
    is_noisy_filter: list[bool] | None,
    predicted_groups: list[str] | None,
) -> pd.DataFrame:
    """Subset one eval pool with optional misclassification / column filters."""
    if df.empty or "true_group" not in df.columns:
        return pd.DataFrame()

    mask = df["true_group"] == true_group
    if only_misclassified and "correct" in df.columns:
        mask &= ~df["correct"]
    if is_noisy_filter is not None and "is_noisy" in df.columns:
        mask &= df["is_noisy"].isin(is_noisy_filter)
    if predicted_groups and "predicted_group" in df.columns:
        mask &= df["predicted_group"].isin(predicted_groups)
    return df.loc[mask].copy()


def prepare_pool_display_table(df: pd.DataFrame, signal: str) -> pd.DataFrame:
    """Table for Streamlit: readable ``correct`` and stable column order."""
    if df.empty:
        return df

    cols = [c for c in POOL_TABLE_COLUMNS if c in df.columns]
    if signal in df.columns and signal not in cols:
        cols.append(signal)

    out = df[cols].copy()
    if "correct" in out.columns:
        out["correct"] = out["correct"].map({True: "yes", False: "no"})
    if "is_test_split" in out.columns:
        out["is_test_split"] = out["is_test_split"].map({True: "test", False: "train"})
    if "is_noisy" in out.columns:
        out["is_noisy"] = out["is_noisy"].map({True: "yes", False: "no"})
    return out


def format_pool_stats_line(stats: dict[str, int | float | None], *, signal: str) -> str:
    """One-line summary: pool size, correct vs misclassified, train/test split."""
    n_pool = int(stats["n_pool"])
    n_correct = int(stats["n_correct"])
    n_fail = int(stats["n_misclassified"])
    acc = stats["accuracy"]
    acc_s = f"{acc:.1%}" if acc is not None else "—"
    parts = [
        f"**{n_pool}** in pool",
        f"**{n_correct}** classified correctly",
        f"**{n_fail}** misclassified",
        f"accuracy **{acc_s}**",
    ]
    if stats.get("n_train") or stats.get("n_test"):
        parts.append(
            f"split: **{stats['n_train']}** train / **{stats['n_test']}** test (for fitting)"
        )
    parts.append(f"score column = `{signal}`")
    return " · ".join(parts)


def architecture_label_to_folder_key(architecture_label: str) -> str | None:
    for key, cfg in ARCHITECTURES.items():
        if cfg["name"] == architecture_label:
            return key
    for key, name in LEGACY_ARCHITECTURE_LABELS.items():
        if name == architecture_label:
            return key
    return None


def resolve_validation_run_dir(
    project_root: Path,
    architecture_label: str,
    sweep_kind: str,
    x_value: float | int,
) -> Path | None:
    """Map human architecture label + sweep point → ``results/validation/*_sweep/<folder>``."""
    key = architecture_label_to_folder_key(architecture_label)
    if key is None:
        return None

    if sweep_kind == "dataset_size":
        folder_name = f"{key}_size{int(x_value)}"
        sweep_dir = project_root / "results" / "validation" / "dataset_size_sweep"
    elif sweep_kind == "label_noise":
        folder_name = f"{key}_noise{int(x_value)}"
        sweep_dir = project_root / "results" / "validation" / "label_noise_sweep"
    else:
        return None

    run_dir = sweep_dir / folder_name
    return run_dir if run_dir.is_dir() else None


def _seed_from_run_dir(run_dir: Path) -> int:
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
            return int(summary.get("config", {}).get("seed", 42))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 42


def enrich_per_sample_with_predictions(
    df: pd.DataFrame,
    signal: str,
    *,
    seed: int,
    train_fraction: float = 0.5,
) -> pd.DataFrame:
    """Add ``predicted_group``, ``correct``, ``is_test_split`` using a single-signal 3-way clf."""
    from uqlab.evaluation.classification.evaluation import predict_eval_groups_single_signal

    if signal not in df.columns or "group" not in df.columns:
        return df.copy()

    out = df.copy()
    group_ids = out["group"].map(GROUP_STR_TO_INT)
    if group_ids.isna().any():
        unknown = sorted(out.loc[group_ids.isna(), "group"].dropna().unique().tolist())
        st.warning(
            "Cannot classify: unknown `group` values "
            f"{unknown}. Expected clean / aleatoric_like / epistemic_like."
        )
        return out

    labels = torch.as_tensor(group_ids.astype(int).tolist(), dtype=torch.long)
    values = torch.as_tensor(out[signal].astype(float).tolist(), dtype=torch.float32)

    pred_ids, is_test = predict_eval_groups_single_signal(
        values,
        labels,
        seed=seed,
        train_fraction=train_fraction,
    )
    out["predicted_group"] = [GROUP_INT_TO_STR.get(int(i), str(i)) for i in pred_ids.tolist()]
    out["true_group"] = out["group"]
    out["correct"] = out["predicted_group"] == out["true_group"]
    out["is_test_split"] = is_test.numpy()
    return out


def failure_table(
    df: pd.DataFrame,
    true_group: str,
    signal: str,
) -> pd.DataFrame:
    """Misclassified samples in ``true_group`` (backward-compatible helper)."""
    sub = filter_eval_pool(
        df,
        true_group,
        only_misclassified=True,
        is_noisy_filter=None,
        predicted_groups=None,
    )
    if sub.empty or signal not in sub.columns:
        return sub
    sort_cols: list[str] = []
    if "is_noisy" in sub.columns:
        sort_cols.append("is_noisy")
    sort_cols.append(signal)
    return sub.sort_values(sort_cols, ascending=[False] * len(sort_cols))


def _metrics_signal_uncertainty_caption(
    metrics_df: pd.DataFrame,
    *,
    architecture: str,
    signal: str,
    x_col: str,
    sweep_point: float | int,
    n_eval_samples: int | None,
) -> str:
    """Explain col-1 curve values at the selected sweep point."""
    row = metrics_df[
        (metrics_df["architecture"] == architecture) & (metrics_df[x_col] == sweep_point)
    ]
    parts = [
        "Curve = **pool means** over all eval samples at each sweep point (`metrics.csv`, MC dropout)."
    ]
    if n_eval_samples is not None:
        parts.append(f"This run: **{n_eval_samples}** rows in `per_sample_signals.csv`.")
    if row.empty:
        return " ".join(parts)
    r = row.iloc[0]
    for key, label in (
        (f"{signal}_mean_epistemic", "mean epistemic"),
        (f"{signal}_mean_aleatoric", "mean aleatoric"),
        ("mean_epistemic_uncertainty", "arch mean epistemic"),
        ("mean_aleatoric_uncertainty", "arch mean aleatoric"),
    ):
        if key in r.index and pd.notna(r[key]):
            parts.append(f"{label}=**{float(r[key]):.3f}**")
    return " ".join(parts)


def _plot_arch_metrics_sweep(
    arch_df: pd.DataFrame,
    x_col: str,
    *,
    y_cols: list[tuple[str, str, str]],
    title: str,
    highlight_x: float | int | None,
    y_range: tuple[float, float] = (0.0, 2.5),
) -> go.Figure | None:
    """Shared sweep plot helper (solid uncertainty lines + accuracy on secondary y)."""
    if arch_df.empty or x_col not in arch_df.columns:
        return None

    arch_df = arch_df.sort_values(x_col)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for col_name, label, color in y_cols:
        if col_name not in arch_df.columns:
            continue
        valid = arch_df[[x_col, col_name]].dropna()
        if valid.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=valid[x_col],
                y=valid[col_name],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=7),
            ),
            secondary_y=False,
        )

    if "accuracy" in arch_df.columns:
        valid_acc = arch_df[[x_col, "accuracy"]].dropna()
        if not valid_acc.empty:
            fig.add_trace(
                go.Scatter(
                    x=valid_acc[x_col],
                    y=valid_acc["accuracy"],
                    mode="lines+markers",
                    name="Accuracy",
                    line=dict(color=ACCURACY_COLOR, width=2, dash="dot"),
                    marker=dict(size=6, symbol="diamond"),
                ),
                secondary_y=True,
            )

    if highlight_x is not None:
        fig.add_vline(
            x=float(highlight_x),
            line_width=1,
            line_dash="dash",
            line_color="gray",
            opacity=0.7,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        height=280,
        margin=dict(t=40, b=32, l=48, r=24),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=11),
            bgcolor="rgba(0, 0, 0, 0)",
            bordercolor="rgba(0, 0, 0, 0)",
            borderwidth=0
        ),
    )
    fig.update_yaxes(title_text="Uncertainty", range=list(y_range), secondary_y=False)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], secondary_y=True)
    fig.update_xaxes(title_text=x_col.replace("_", " ").title())
    return fig


def plot_architecture_row_sweep(
    metrics_df: pd.DataFrame,
    *,
    architecture: str,
    x_col: str,
    highlight_x: float | int | None = None,
) -> go.Figure | None:
    """Row 1: aggregate mean epistemic / aleatoric uncertainty for the architecture."""
    arch_df = metrics_df[metrics_df["architecture"] == architecture].copy()
    cols = [
        (c, lbl, clr)
        for c, lbl, clr in zip(
            ARCHITECTURE_ROW.columns,
            ARCHITECTURE_ROW.labels,
            ARCHITECTURE_ROW.colors,
        )
    ]
    return _plot_arch_metrics_sweep(
        arch_df,
        x_col,
        y_cols=cols,
        title="Architecture (pool means)",
        highlight_x=highlight_x,
        y_range=ARCHITECTURE_ROW.y_range,
    )


def plot_single_arch_signal_sweep(
    metrics_df: pd.DataFrame,
    *,
    architecture: str,
    signal: str,
    x_col: str,
    highlight_x: float | int | None = None,
) -> go.Figure | None:
    """Signal row col 1: epistemic + aleatoric pool means for this signal + accuracy."""
    arch_df = metrics_df[metrics_df["architecture"] == architecture].copy()
    spec = UNCERTAINTY_DECOMPOSITION
    cols = [
        (c, lbl, clr)
        for c, lbl, clr in zip(
            resolve_columns(spec, signal=signal, auroc_type="epistemic"),
            spec.labels,
            spec.colors,
        )
    ]
    sig_lbl = SIGNAL_LABELS.get(signal, signal)
    return _plot_arch_metrics_sweep(
        arch_df,
        x_col,
        y_cols=cols,
        title=sig_lbl,
        highlight_x=highlight_x,
        y_range=spec.y_range,
    )


def _auroc_for_signal_at_point(
    metrics_df: pd.DataFrame,
    *,
    architecture: str,
    signal: str,
    sweep_type: str,
    x_col: str,
    sweep_point: float | int,
) -> dict[str, float | None]:
    """Run-level AUROC columns from metrics.csv at the selected sweep point."""
    auroc_type = sweep_to_auroc_type(sweep_type)
    other = "aleatoric" if auroc_type == "epistemic" else "epistemic"
    row = metrics_df[
        (metrics_df["architecture"] == architecture) & (metrics_df[x_col] == sweep_point)
    ]
    if row.empty:
        return {auroc_type: None, other: None}
    r = row.iloc[0]
    return {
        auroc_type: float(r[f"{signal}_{auroc_type}_auroc"])
        if f"{signal}_{auroc_type}_auroc" in r and pd.notna(r[f"{signal}_{auroc_type}_auroc"])
        else None,
        other: float(r[f"{signal}_{other}_auroc"])
        if f"{signal}_{other}_auroc" in r and pd.notna(r[f"{signal}_{other}_auroc"])
        else None,
    }


def _append_auroc_caption(sweep_type: str, auroc: dict[str, float | None]) -> None:
    primary = sweep_to_auroc_type(sweep_type)
    other = "aleatoric" if primary == "epistemic" else "epistemic"
    parts = []
    if auroc.get(primary) is not None:
        parts.append(f"{primary} AUROC={auroc[primary]:.3f}")
    if auroc.get(other) is not None:
        parts.append(f"{other} AUROC={auroc[other]:.3f}")
    if parts:
        st.caption(" · ".join(parts) + " (from metrics.csv at this sweep point)")


def _render_pool_decision_column(
    enriched: pd.DataFrame,
    true_group: str,
    signal: str,
    *,
    pool_label: str,
    widget_key: str,
    default_only_misclassified: bool = True,
) -> None:
    """Interactive eval-pool table with filters and explicit correct/total counts."""
    if "predicted_group" not in enriched.columns:
        st.warning("Classifier did not run (check `group` values in per-sample CSV).")
        return

    stats = pool_classification_stats(enriched, true_group)
    st.markdown(f"**{pool_label}**")
    st.caption(format_pool_stats_line(stats, signal=signal))

    pool = enriched.loc[enriched["true_group"] == true_group] if "true_group" in enriched.columns else enriched
    only_fail = st.checkbox(
        "Only misclassified",
        value=default_only_misclassified,
        key=f"{widget_key}_only_fail",
    )
    noisy_opts: list[bool] = []
    if "is_noisy" in pool.columns:
        noisy_opts = sorted(pool["is_noisy"].dropna().unique().tolist())
    is_noisy_filter: list[bool] | None = None
    if noisy_opts:
        labels = {False: "not noisy", True: "noisy"}
        picked = st.multiselect(
            "Filter is_noisy",
            options=noisy_opts,
            format_func=lambda v: labels.get(v, str(v)),
            default=noisy_opts,
            key=f"{widget_key}_noisy",
        )
        is_noisy_filter = picked if picked != noisy_opts else None

    pred_opts = (
        sorted(pool["predicted_group"].dropna().unique().tolist())
        if "predicted_group" in pool.columns
        else []
    )
    predicted_groups: list[str] | None = None
    if pred_opts:
        picked_pred = st.multiselect(
            "Filter predicted_group",
            options=pred_opts,
            default=pred_opts,
            key=f"{widget_key}_pred",
        )
        predicted_groups = picked_pred if picked_pred != pred_opts else None

    filtered = filter_eval_pool(
        enriched,
        true_group,
        only_misclassified=only_fail,
        is_noisy_filter=is_noisy_filter,
        predicted_groups=predicted_groups,
    )
    if signal in filtered.columns:
        sort_cols: list[str] = []
        if "is_noisy" in filtered.columns:
            sort_cols.append("is_noisy")
        sort_cols.append(signal)
        filtered = filtered.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    shown = len(filtered)
    st.caption(f"Showing **{shown}** / **{stats['n_pool']}** rows")
    if shown == 0:
        if only_fail and stats["n_misclassified"] == 0:
            st.success("No misclassifications in this pool.")
        else:
            st.info("No rows match the current filters.")
        return

    st.dataframe(
        prepare_pool_display_table(filtered, signal),
        use_container_width=True,
        height=360,
        hide_index=True,
    )


def _render_signal_row(
    *,
    plot_df: pd.DataFrame,
    per_sample: pd.DataFrame | None,
    architecture: str,
    signal: str,
    sweep_type: str,
    x_col: str,
    sweep_point: float | int,
    seed: int,
    row_prefix: str,
) -> None:
    """One signal: col1 uncertainty sweep, col2/col3 misclassification tables."""
    sig_lbl = SIGNAL_LABELS.get(signal, signal)
    st.markdown(f"##### {sig_lbl} (`{signal}`)")
    col_plot, col_alea, col_epis = st.columns([1.1, 1, 1])

    with col_plot:
        fig = plot_single_arch_signal_sweep(
            plot_df,
            architecture=architecture,
            signal=signal,
            x_col=x_col,
            highlight_x=sweep_point,
        )
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"{row_prefix}_plot_{signal}")
        else:
            st.info("No sweep metrics for this signal.")
        _append_auroc_caption(
            sweep_type,
            _auroc_for_signal_at_point(
                plot_df,
                architecture=architecture,
                signal=signal,
                sweep_type=sweep_type,
                x_col=x_col,
                sweep_point=sweep_point,
            ),
        )
        n_eval = len(per_sample) if per_sample is not None else None
        st.caption(
            _metrics_signal_uncertainty_caption(
                plot_df,
                architecture=architecture,
                signal=signal,
                x_col=x_col,
                sweep_point=sweep_point,
                n_eval_samples=n_eval,
            )
        )

    if per_sample is None or signal not in per_sample.columns:
        with col_alea:
            st.info("Per-sample CSV required for failure tables.")
        with col_epis:
            st.info("—")
        return

    enriched = enrich_per_sample_with_predictions(per_sample, signal, seed=seed)
    with col_alea:
        _render_pool_decision_column(
            enriched,
            "aleatoric_like",
            signal,
            pool_label="Aleatoric pool",
            widget_key=f"{row_prefix}_alea_{signal}",
        )
    with col_epis:
        _render_pool_decision_column(
            enriched,
            "epistemic_like",
            signal,
            pool_label="Epistemic pool",
            widget_key=f"{row_prefix}_epis_{signal}",
        )


def render_visualize_main(
    metrics_df: pd.DataFrame,
    *,
    project_root: Path,
    sweep_type: str,
    x_col: str,
    architecture: str,
    sweep_point: float | int,
    selected_signals: list[str],
) -> None:
    """
    Primary Visualize layout: row 1 architecture, then one row per signal.

    Column 1 = uncertainty curves (solid). Columns 2–3 = per-sample decisions
    that drove misclassification in aleatoric / epistemic eval pools.
    """
    if metrics_df.empty:
        st.warning("No metrics loaded for this sweep.")
        return

    plot_df = metrics_df
    run_dir = resolve_validation_run_dir(
        project_root, architecture, sweep_type, sweep_point
    )
    if run_dir is None:
        st.warning(
            f"No run folder for **{architecture}** at {x_col}={sweep_point}. "
            "Run validation experiments for this point first."
        )
    else:
        try:
            rel = run_dir.relative_to(project_root)
        except ValueError:
            rel = run_dir
        per_sample_path = run_dir / "per_sample_signals.csv"
        if per_sample_path.is_file():
            st.caption(f"Per-sample decisions: `{rel}` / `per_sample_signals.csv`")
        else:
            st.warning(f"`{rel}` has no `per_sample_signals.csv` — tables need a PyTorch run.")

    artifacts = load_run_directory(run_dir) if run_dir else None
    per_sample = (
        load_per_sample_table(run_dir, max_rows=5000)
        if run_dir and (run_dir / "per_sample_signals.csv").is_file()
        else None
    )
    seed = _seed_from_run_dir(run_dir) if run_dir else 42

    # --- Row 1: architecture-level uncertainty ---
    st.markdown("#### Row 1 — Architecture (aggregate pool uncertainty)")
    ranked = get_row3_signals(plot_df, sweep_type=sweep_type)
    probe_signal = ranked[0][0] if ranked else (selected_signals[0] if selected_signals else "mutual_info")

    col_r1_plot, col_r1_alea, col_r1_epis = st.columns([1.1, 1, 1])
    with col_r1_plot:
        fig = plot_architecture_row_sweep(
            plot_df,
            architecture=architecture,
            x_col=x_col,
            highlight_x=sweep_point,
        )
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key="hv_v2_arch_row_plot")
        else:
            st.info("No architecture-level metrics.")
        n_eval = len(per_sample) if per_sample is not None else None
        st.caption(
            _metrics_signal_uncertainty_caption(
                plot_df,
                architecture=architecture,
                signal=probe_signal,
                x_col=x_col,
                sweep_point=sweep_point,
                n_eval_samples=n_eval,
            )
        )

    if per_sample is not None and probe_signal in per_sample.columns:
        enriched_probe = enrich_per_sample_with_predictions(per_sample, probe_signal, seed=seed)
        with col_r1_alea:
            st.caption(f"Probe signal: `{probe_signal}` (top mean AUROC)")
            _render_pool_decision_column(
                enriched_probe,
                "aleatoric_like",
                probe_signal,
                pool_label="Aleatoric pool",
                widget_key="hv_v2_r1_alea",
            )
        with col_r1_epis:
            _render_pool_decision_column(
                enriched_probe,
                "epistemic_like",
                probe_signal,
                pool_label="Epistemic pool",
                widget_key="hv_v2_r1_epis",
            )
    else:
        with col_r1_alea:
            st.info("Per-sample CSV needed for decision tables.")
        with col_r1_epis:
            st.info("—")

    with st.expander("How uncertainty (col 1) vs classifier (cols 2–3) work", expanded=False):
        st.markdown(CLASSIFICATION_METHOD_MD)

    st.markdown("---")
    st.markdown("#### Signal rows — uncertainty vs pool decisions")
    st.caption(
        "Col 1: epistemic + aleatoric **mean uncertainty** per sweep point. "
        "Cols 2–3: per-sample 3-way linear classifier on this signal; use filters to explore "
        "misclassified rows (`correct` = yes/no vs ground-truth `group`)."
    )

    for signal in selected_signals:
        _render_signal_row(
            plot_df=plot_df,
            per_sample=per_sample,
            architecture=architecture,
            signal=signal,
            sweep_type=sweep_type,
            x_col=x_col,
            sweep_point=sweep_point,
            seed=seed,
            row_prefix="hv_v2",
        )

    if artifacts and artifacts.eval_sizes:
        with st.expander("Eval pool sizes (this run)", expanded=False):
            st.json(artifacts.eval_sizes)


def render_signal_diagnostic_panel(
    metrics_df: pd.DataFrame,
    *,
    project_root: Path,
    sweep_type: str,
    x_col: str,
    pytorch_only: bool = True,
) -> None:
    """Backward-compatible wrapper with internal architecture/sweep selectors."""
    if metrics_df.empty:
        st.warning("No metrics loaded for this sweep.")
        return

    plot_df = metrics_df
    if pytorch_only and "source" in plot_df.columns:
        pytorch = plot_df[plot_df["source"] == "pytorch_validation"]
        if not pytorch.empty:
            plot_df = pytorch

    arch_options = sorted(plot_df["architecture"].dropna().unique().tolist())
    if not arch_options:
        st.warning("No architectures in metrics CSV.")
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        architecture = st.selectbox("Architecture", arch_options, key="sig_diag_architecture")
    arch_metrics = plot_df[plot_df["architecture"] == architecture].copy()
    if x_col not in arch_metrics.columns:
        st.error(f"Missing x column `{x_col}` in metrics.")
        return

    x_values = sorted(arch_metrics[x_col].dropna().unique().tolist())
    if not x_values:
        st.warning("No sweep points for this architecture.")
        return

    with c2:
        sweep_point = st.selectbox("Sweep point", x_values, key="sig_diag_sweep_point")
    with c3:
        signal_options = [s for s in FAST_PILOT_SIGNAL_NAMES if f"{s}_mean_epistemic" in plot_df.columns]
        if not signal_options:
            signal_options = list(FAST_PILOT_SIGNAL_NAMES)
        selected_signals = st.multiselect(
            "Signals",
            signal_options,
            default=signal_options,
            key="sig_diag_signals",
        )

    render_visualize_main(
        plot_df,
        project_root=project_root,
        sweep_type=sweep_type,
        x_col=x_col,
        architecture=architecture,
        sweep_point=sweep_point,
        selected_signals=selected_signals,
    )
