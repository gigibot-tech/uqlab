"""
Modular 3-line sweep plot: X = swept param, left Y = signal means per eval pack, right Y = accuracy.

Pool lines are **config-implicit**: primary pool follows the swept axis (aleatoric for label-noise,
epistemic for under-train); optional dashed mirror only when that pool's mean column exists in artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from uqlab.evaluation.pipeline.sweep_plot_pools import (
    list_plottable_signals_for_sweep,
    mirror_omitted_note,
    pool_expectations_from_data_config,
    primary_pool_for_sweep,
    resolve_sweep_plot_traces,
    sweep_pool_caption,
)
from uqlab.shared.config.signals import plot_default_signal_for_sweep
from uqlab.notebook_support.metric_specs import ACCURACY_COLOR
from uqlab.run_artifacts import metrics_row_from_run
from uqlab.runtime_paths import resolve_experiment_results_dir
from uqlab.shared.notebook_utils.signals import SIGNAL_LABELS

SWEEP_KIND_LABEL_NOISE = "label_noise"
SWEEP_KIND_DATASET_SIZE = "dataset_size"

_X_LABELS = {
    "noise_percent": "Label noise (%)",
    "under_train_per_class": "Under-train per class",
}

# Secondary sweep dimensions (Z / facet) — hold constant while X varies.
FACET_PARAM_LABELS: dict[str, str] = {
    "learning_rate": "Learning rate",
    "epochs": "Epochs",
    "dropout": "Dropout",
    "architecture": "Architecture",
}
FACET_CANDIDATES: tuple[str, ...] = tuple(FACET_PARAM_LABELS.keys())


def run_ids_for_experiments(
    experiments: list[dict[str, Any]],
    *,
    experiments_dir: Path | None = None,
) -> list[str]:
    """Map API experiment records to on-disk run folders with result artifacts."""
    run_ids: list[str] = []
    seen: set[str] = set()
    for exp in experiments:
        eid = str(exp.get("id") or "")
        if not eid:
            continue
        if experiments_dir is not None:
            results_dir = experiments_dir / eid / "results"
        else:
            results_dir = resolve_experiment_results_dir(
                eid,
                results_path=exp.get("results_path"),
            )
        if not (
            (results_dir / "summary.json").is_file()
            or (results_dir / "results.pt").is_file()
        ):
            continue
        run_id = results_dir.parent.name
        if run_id in seen:
            continue
        seen.add(run_id)
        run_ids.append(run_id)
    return run_ids


@dataclass(frozen=True)
class SweepLinePlotPayload:
    """JSON-serializable plot description for Plotly (or any line chart)."""

    sweep_kind: str
    x_col: str
    x_label: str
    signal: str
    signal_label: str
    default_signal: str
    available_signals: list[str]
    y_left_title: str
    y_right_title: str
    traces: list[dict[str, Any]]
    points: int
    facet_filters: dict[str, Any] | None = None
    facet_options: dict[str, list[Any]] | None = None
    pool_caption: str = ""
    primary_pool: str = ""
    has_mirror_line: bool = False
    mirror_note: str | None = None
    plot_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sweep_kind": self.sweep_kind,
            "x_col": self.x_col,
            "x_label": self.x_label,
            "signal": self.signal,
            "signal_label": self.signal_label,
            "default_signal": self.default_signal,
            "available_signals": self.available_signals,
            "y_left_title": self.y_left_title,
            "y_right_title": self.y_right_title,
            "traces": self.traces,
            "points": self.points,
            "facet_filters": self.facet_filters or {},
            "facet_options": self.facet_options or {},
            "pool_caption": self.pool_caption,
            "primary_pool": self.primary_pool,
            "has_mirror_line": self.has_mirror_line,
            "mirror_note": self.mirror_note,
            "plot_config": self.plot_config or {},
        }


def default_signal_for_sweep(sweep_kind: str) -> str:
    """Sensible picker default — aleatoric candidate for noise, epistemic for dataset size."""
    return plot_default_signal_for_sweep(sweep_kind)


def resolve_sweep_trace_xy(
    traces: list[dict[str, Any]],
    trace: dict[str, Any],
) -> tuple[list[float], list[float]]:
    """Align x/y for matplotlib — fill missing x from sibling traces, skip null y."""
    shared_x = next((t.get("x") for t in traces if t.get("x")), None) or []
    x_raw = trace.get("x") or shared_x
    y_raw = trace.get("y") or []
    xs: list[float] = []
    ys: list[float] = []
    for xi, yi in zip(x_raw, y_raw):
        if yi is None:
            continue
        try:
            yf = float(yi)
            if yf != yf:
                continue
        except (TypeError, ValueError):
            continue
        try:
            xs.append(float(xi))
            ys.append(yf)
        except (TypeError, ValueError):
            continue
    return xs, ys


def _run_name_blob(df: pd.DataFrame) -> str:
    if "run_name" not in df.columns:
        return ""
    return " ".join(str(x) for x in df["run_name"].dropna())


def _unique_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].nunique(dropna=True))


def sweep_kind_from_group(group: dict[str, Any]) -> str | None:
    """
    Best-effort sweep kind from API grouping metadata (name pattern / swept_param).

    Returns ``label_noise``, ``dataset_size``, or ``None``.
    """
    param = str(group.get("swept_param", "")).lower()
    if any(tok in param for tok in ("under", "dataset", "epis", "data.under_train")):
        return SWEEP_KIND_DATASET_SIZE
    if any(tok in param for tok in ("noise", "alea", "aleatoric")):
        return SWEEP_KIND_LABEL_NOISE

    names = " ".join(str(e.get("name", "")) for e in group.get("experiments") or [])
    if "fast_epis" in names or "_under_" in names:
        return SWEEP_KIND_DATASET_SIZE
    if "fast_alea" in names or "_noise_" in names:
        return SWEEP_KIND_LABEL_NOISE
    return None


def infer_sweep_kind(
    df: pd.DataFrame,
    *,
    hint: str | None = None,
) -> str:
    """
    Detect label-noise vs under-train sweep.

    Prefer explicit *hint* (from campaign metadata / user picker), then run names,
    then whichever axis has more unique values in the metrics frame.
    """
    if hint in (SWEEP_KIND_LABEL_NOISE, SWEEP_KIND_DATASET_SIZE):
        x_col = resolve_x_col(df, hint)
        if _unique_count(df, x_col) > 1:
            return hint

    names = _run_name_blob(df)
    if "fast_epis" in names or "_under_" in names:
        return SWEEP_KIND_DATASET_SIZE
    if "fast_alea" in names or "_noise_" in names:
        return SWEEP_KIND_LABEL_NOISE

    noise_u = _unique_count(df, "noise_percent")
    under_u = _unique_count(df, "under_train_per_class")

    if noise_u > 1 and under_u > 1:
        return SWEEP_KIND_DATASET_SIZE if under_u >= noise_u else SWEEP_KIND_LABEL_NOISE
    if under_u > 1:
        return SWEEP_KIND_DATASET_SIZE
    if noise_u > 1:
        return SWEEP_KIND_LABEL_NOISE

    if hint in (SWEEP_KIND_LABEL_NOISE, SWEEP_KIND_DATASET_SIZE):
        return hint

    return SWEEP_KIND_LABEL_NOISE


def _load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.yaml"
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _enrich_metrics_row(results_dir: Path) -> dict[str, Any]:
    """One sweep point: signal means + accuracy + sweep keys from config."""
    row = metrics_row_from_run(results_dir)
    run_dir = results_dir.parent
    row["run_id"] = run_dir.name
    cfg = _load_config(run_dir)
    data = cfg.get("data") or {}
    model = cfg.get("model") or {}
    training = cfg.get("training") or {}
    seed = int(cfg.get("seed") or 42)

    expectations = pool_expectations_from_data_config(data, seed=seed)
    row["epistemic_pool_expected"] = expectations.epistemic_pool_expected
    row["aleatoric_pool_expected"] = expectations.aleatoric_pool_expected

    arch = model.get("architecture") or "unknown"
    row["architecture"] = str(arch)

    if training.get("learning_rate") is not None:
        row["learning_rate"] = float(training["learning_rate"])
    if training.get("epochs") is not None:
        row["epochs"] = int(training["epochs"])
    if model.get("dropout") is not None:
        row["dropout"] = float(model["dropout"])

    alea = data.get("aleatoric_noise_percentage")
    if alea is not None:
        row["noise_percent"] = float(alea)

    under = data.get("under_train_per_class")
    if under is not None:
        row["under_train_per_class"] = int(under)
        row["dataset_size"] = int(under)

    meta_path = results_dir.parent / "meta.json"
    if meta_path.is_file():
        import json

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            row["run_name"] = meta.get("name")
        except Exception:
            pass

    return row


def build_sweep_metrics_frame(
    run_ids: list[str],
    experiments_dir: Path,
) -> pd.DataFrame:
    """Aggregate completed runs in a sweep group into one metrics frame."""
    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        results_dir = experiments_dir / run_id / "results"
        if not (results_dir / "summary.json").is_file() and not (results_dir / "results.pt").is_file():
            continue
        rows.append(_enrich_metrics_row(results_dir))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def detect_facet_columns(df: pd.DataFrame, x_col: str) -> dict[str, list[Any]]:
    """
    Columns that vary across runs but are not the primary X axis.

    Used as the Z / facet dimension: pick one value to slice a clean 3-line plot.
    """
    facets: dict[str, list[Any]] = {}
    for col in FACET_CANDIDATES:
        if col not in df.columns or col == x_col:
            continue
        series = df[col].dropna()
        if series.nunique() <= 1:
            continue
        facets[col] = sorted(series.unique().tolist(), key=lambda v: (isinstance(v, str), v))
    return facets


def filter_metrics_frame(
    df: pd.DataFrame,
    facet_filters: dict[str, Any] | None,
) -> pd.DataFrame:
    """Keep rows matching facet slice (omit keys set to None or ``\"all\"``)."""
    if not facet_filters or df.empty:
        return df
    out = df.copy()
    for col, value in facet_filters.items():
        if value is None or value == "all" or col not in out.columns:
            continue
        out = out[out[col] == value]
    return out


def run_ids_from_metrics_frame(df: pd.DataFrame) -> list[str]:
    if df.empty or "run_id" not in df.columns:
        return []
    return [str(r) for r in df["run_id"].dropna().tolist()]


def resolve_x_col(df: pd.DataFrame, sweep_kind: str) -> str:
    if sweep_kind == SWEEP_KIND_LABEL_NOISE and "noise_percent" in df.columns:
        return "noise_percent"
    if sweep_kind == SWEEP_KIND_DATASET_SIZE:
        for col in ("under_train_per_class", "dataset_size"):
            if col in df.columns:
                return col
    for col in ("noise_percent", "under_train_per_class", "dataset_size"):
        if col in df.columns:
            return col
    raise ValueError("No sweep x-axis column in metrics frame")


def list_plottable_signals(df: pd.DataFrame, sweep_kind: str | None = None) -> list[str]:
    """Signals with primary-pool mean column for the sweep axis (config-implicit)."""
    if sweep_kind is None:
        sweep_kind = infer_sweep_kind(df)
    return list_plottable_signals_for_sweep(df, sweep_kind)


def _pick_signal(df: pd.DataFrame, sweep_kind: str, signal: Optional[str]) -> str:
    """Select which uncertainty signal to plot, with smart defaults."""
    from uqlab.evaluation.signals.catalog import normalize_signal_id

    available = list_plottable_signals(df, sweep_kind)
    if not available:
        raise ValueError(
            "No plottable signals (need results.pt with per-pool means). "
            "Re-run completed experiments or wait for sweep to finish."
        )
    if signal:
        normalized = normalize_signal_id(signal)
        if normalized in available:
            return normalized
        if signal in available:
            return signal
    default = default_signal_for_sweep(sweep_kind)
    if default in available:
        return default
    return available[0]


def _build_uncertainty_traces(
    plot_df: pd.DataFrame,
    x_vals: list[float],
    trace_specs,
) -> list[dict[str, Any]]:
    """
    Build trace data for uncertainty lines (primary + optional mirror pool).
    
    Each trace represents one uncertainty pool's mean values across the sweep.
    Primary pool is solid line, mirror pool (if present) is dashed.
    """
    traces: list[dict[str, Any]] = []
    
    for spec in trace_specs:
        y_vals = plot_df[spec.column].tolist()
        traces.append(
            {
                "name": spec.label,
                "column": spec.column,
                "x": x_vals,
                "y": [None if pd.isna(v) else float(v) for v in y_vals],
                "yaxis": "left",  # Uncertainty on left Y-axis
                "color": spec.color,
                "dash": spec.dash,  # "solid" for primary, "dash" for mirror
            }
        )
    
    return traces


def _build_accuracy_trace(
    plot_df: pd.DataFrame,
    x_vals: list[float],
) -> dict[str, Any]:
    """
    Build trace data for accuracy line (right Y-axis, dotted).
    
    Accuracy shows model performance at each sweep point, providing
    context for how uncertainty metrics relate to classification quality.
    """
    return {
        "name": "Accuracy",
        "column": "accuracy",
        "x": x_vals,
        "y": [None if pd.isna(v) else float(v) for v in plot_df["accuracy"].tolist()],
        "yaxis": "right",  # Accuracy on right Y-axis
        "color": ACCURACY_COLOR,
        "dash": "dot",  # Dotted line distinguishes from uncertainty
    }


def _package_plot_payload(
    *,
    full_df: pd.DataFrame,
    plot_df: pd.DataFrame,
    run_ids: list[str],
    sweep_kind: str,
    x_col: str,
    chosen: str,
    trace_specs,
    traces: list[dict[str, Any]],
    architecture: str | None,
    facet_filters: dict[str, Any] | None,
    facet_options: dict[str, list[Any]],
) -> SweepLinePlotPayload:
    """
    Package all plot data and metadata into final payload for rendering.
    
    This creates the complete specification needed to render the 3-line plot
    in any visualization library (Plotly, matplotlib, etc.).
    """
    available = list_plottable_signals(full_df, sweep_kind)
    active_facets = {
        k: v for k, v in (facet_filters or {}).items()
        if v is not None and v != "all"
    }
    primary = primary_pool_for_sweep(sweep_kind)
    has_mirror = any(not spec.primary for spec in trace_specs)
    mirror_note = mirror_omitted_note(full_df, chosen, sweep_kind) if not has_mirror else None
    signal_label = SIGNAL_LABELS.get(chosen, chosen)
    pool_label = "aleatoric_like" if primary == "aleatoric" else "epistemic_like"
    
    plot_config = _build_plot_config(
        run_ids=run_ids,
        sweep_kind=sweep_kind,
        x_col=x_col,
        x_label=_X_LABELS.get(x_col, x_col.replace("_", " ").title()),
        signal=chosen,
        signal_label=signal_label,
        trace_specs=trace_specs,
        plot_df=plot_df,
        architecture=architecture,
        facet_filters=active_facets,
        facet_options=facet_options or {},
        primary=primary,
        has_mirror=has_mirror,
        mirror_note=mirror_note,
        available_signals=available,
    )
    
    return SweepLinePlotPayload(
        sweep_kind=sweep_kind,
        x_col=x_col,
        x_label=_X_LABELS.get(x_col, x_col.replace("_", " ").title()),
        signal=chosen,
        signal_label=signal_label,
        default_signal=default_signal_for_sweep(sweep_kind),
        available_signals=available,
        y_left_title=f"{signal_label} (mean on {pool_label} eval pack)",
        y_right_title="Accuracy",
        traces=traces,
        points=len(plot_df),
        facet_filters=active_facets or None,
        facet_options=facet_options or None,
        pool_caption=sweep_pool_caption(
            sweep_kind,
            has_mirror=has_mirror,
            mirror_note=mirror_note,
        ),
        primary_pool=primary,
        has_mirror_line=has_mirror,
        mirror_note=mirror_note,
        plot_config=plot_config,
    )


def _build_plot_config(
    *,
    run_ids: list[str],
    sweep_kind: str,
    x_col: str,
    x_label: str,
    signal: str,
    signal_label: str,
    trace_specs,
    plot_df: pd.DataFrame,
    architecture: str | None,
    facet_filters: dict[str, Any],
    facet_options: dict[str, list[Any]],
    primary: str,
    has_mirror: bool,
    mirror_note: str | None,
    available_signals: list[str],
) -> dict[str, Any]:
    """Exact, JSON-serializable spec for reproducing this sweep line plot."""
    y_columns = [
        {
            "column": spec.column,
            "pool": spec.pool,
            "role": "primary" if spec.primary else "mirror",
            "dash": spec.dash,
            "label": spec.label,
        }
        for spec in trace_specs
    ]

    sweep_points: list[dict[str, Any]] = []
    if "run_id" in plot_df.columns:
        for _, row in plot_df.iterrows():
            point: dict[str, Any] = {
                "run_id": str(row["run_id"]),
                x_col: None if pd.isna(row[x_col]) else float(row[x_col]),
            }
            for spec in trace_specs:
                val = row.get(spec.column)
                point[spec.column] = None if pd.isna(val) else float(val)
            if "accuracy" in plot_df.columns:
                acc = row.get("accuracy")
                point["accuracy"] = None if pd.isna(acc) else float(acc)
            sweep_points.append(point)

    run_ids_plotted = (
        [str(r) for r in plot_df["run_id"].dropna().tolist()]
        if "run_id" in plot_df.columns
        else []
    )

    return {
        "sweep_kind": sweep_kind,
        "x_axis": {
            "column": x_col,
            "label": x_label,
        },
        "signal": {
            "id": signal,
            "label": signal_label,
        },
        "pools": {
            "primary": primary,
            "has_mirror_line": has_mirror,
            "mirror_note": mirror_note,
            "y_columns": y_columns,
        },
        "right_y": {
            "column": "accuracy",
            "label": "Accuracy",
            "dash": "dot",
        },
        "facet_filters": facet_filters,
        "facet_options": facet_options,
        "architecture_filter": architecture,
        "run_ids_requested": list(run_ids),
        "run_ids_plotted": run_ids_plotted,
        "points": len(plot_df),
        "available_signals": available_signals,
        "sweep_points": sweep_points,
    }


def build_sweep_line_plot(
    run_ids: list[str],
    experiments_dir: Path,
    *,
    signal: Optional[str] = None,
    architecture: Optional[str] = None,
    facet_filters: dict[str, Any] | None = None,
    sweep_kind: Optional[str] = None,
) -> SweepLinePlotPayload:
    """
    Build 3-line sweep visualization showing how uncertainty and accuracy change across parameter sweeps.
    
    **The Three Lines:**
    1. **Primary pool uncertainty** (solid line, left Y-axis) - e.g., aleatoric for noise sweeps
    2. **Mirror pool uncertainty** (dashed line, left Y-axis, optional) - e.g., epistemic for noise sweeps
    3. **Model accuracy** (dotted line, right Y-axis) - classification performance
    
    **How Curves Are Generated:**
    Each point on the curve represents one completed experiment run:
    - X-axis: swept parameter value (e.g., 0%, 20%, 40% label noise)
    - Y-left: mean uncertainty from that run's evaluation results
    - Y-right: accuracy from that run's evaluation results
    
    The curves connect these pre-computed per-run metrics, sorted by X-axis value.
    
    **4D Visualization Model:**
    - X = primary swept parameter (noise % or dataset size)
    - Y = pool uncertainty means + accuracy
    - Signal = which uncertainty metric to display (epistemic/aleatoric/mutual_info/etc.)
    - Z = facet dimension (learning_rate/epochs/dropout) - use ``facet_filters`` to slice
    
    Args:
        run_ids: Experiment run IDs to include in sweep
        experiments_dir: Root directory containing experiment folders
        signal: Which uncertainty signal to plot (auto-selected if None)
        architecture: Filter to specific model architecture (optional)
        facet_filters: Hold facet dimensions constant (e.g., {"learning_rate": 0.001})
        sweep_kind: "label_noise" or "dataset_size" (auto-detected if None)
        
    Returns:
        SweepLinePlotPayload with traces ready for Plotly rendering
        
    Raises:
        ValueError: If no completed runs found or facet slice is empty
    """
    # ========== STEP 1: Load and prepare metrics data ==========
    full_df = build_sweep_metrics_frame(run_ids, experiments_dir)
    if full_df.empty:
        raise ValueError("No completed runs with metrics in this sweep group")

    # ========== STEP 2: Determine sweep configuration ==========
    sweep_kind = infer_sweep_kind(full_df, hint=sweep_kind)
    x_col = resolve_x_col(full_df, sweep_kind)
    facet_options = detect_facet_columns(full_df, x_col)

    # ========== STEP 3: Apply filters (facets + architecture) ==========
    df = filter_metrics_frame(full_df, facet_filters)
    if df.empty:
        raise ValueError(
            "No runs match the selected facet slice. "
            "Pick another learning rate / epoch / architecture value."
        )

    if architecture and "architecture" in df.columns:
        arch_df = df[df["architecture"] == architecture].copy()
        if not arch_df.empty:
            df = arch_df

    # ========== STEP 4: Select signal and determine which pools to plot ==========
    chosen = _pick_signal(df, sweep_kind, signal)
    trace_specs = resolve_sweep_plot_traces(chosen, sweep_kind, df)
    
    # ========== STEP 5: Extract plotting data and sort by X-axis ==========
    trace_cols = [spec.column for spec in trace_specs]
    cols = [x_col, *trace_cols]
    if "accuracy" in df.columns:
        cols.append("accuracy")
    plot_df = df[cols].copy()
    plot_df = plot_df.dropna(subset=[x_col]).sort_values(x_col)  # Sort creates the curve!

    if plot_df.empty:
        raise ValueError(f"No numeric sweep points for signal {chosen!r}")

    # ========== STEP 6: Build trace data for uncertainty lines ==========
    x_vals = [float(v) for v in plot_df[x_col].tolist()]
    traces = _build_uncertainty_traces(plot_df, x_vals, trace_specs)
    
    # ========== STEP 7: Add accuracy trace (right Y-axis) ==========
    if "accuracy" in plot_df.columns:
        traces.append(_build_accuracy_trace(plot_df, x_vals))

    # ========== STEP 8: Package metadata and return payload ==========
    return _package_plot_payload(
        full_df=full_df,
        plot_df=plot_df,
        run_ids=run_ids,
        sweep_kind=sweep_kind,
        x_col=x_col,
        chosen=chosen,
        trace_specs=trace_specs,
        traces=traces,
        architecture=architecture,
        facet_filters=facet_filters,
        facet_options=facet_options,
    )
