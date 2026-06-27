"""
Campaign config timeline — shared baseline + step-wise parameter changes.

Builds a narrative: baseline config → run → Δparams → run → Δparams → …
Used by the campaign PDF report and the compact config-timeline figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import yaml

from uqlab.evaluation.reporting.sweep_line_plot import (
    build_sweep_metrics_frame,
    filter_metrics_frame,
    infer_sweep_kind,
    resolve_x_col,
    run_ids_for_experiments,
)
from uqlab.evaluation.reporting.config_diff import (
    ConfigChange,
    find_tracked_differences,
    format_value as _format_value,
    label_key as _label_key,
    shared_config as _shared_config,
    tracked_flat as _tracked_flat,
    value_key as _value_key,
)


@dataclass(frozen=True)
class CampaignStep:
    index: int
    run_id: str
    run_name: str
    x_value: float | None
    x_label: str
    changes_from_prev: tuple[ConfigChange, ...]
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_baseline(self) -> bool:
        return self.index == 0


@dataclass(frozen=True)
class CampaignTimeline:
    title: str
    sweep_kind: str
    x_label: str
    shared_config: dict[str, str]
    steps: tuple[CampaignStep, ...]
    facet_note: str = ""

    @property
    def n_runs(self) -> int:
        return len(self.steps)


def _load_run_config(run_id: str, experiments_dir: Path) -> dict[str, Any]:
    path = experiments_dir / str(run_id) / "config.yaml"
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def _metric_snapshot(row: dict[str, Any], x_col: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if x_col in row and row[x_col] is not None:
        out["x"] = row[x_col]
    if row.get("accuracy") is not None:
        out["accuracy"] = float(row["accuracy"])
    for col, val in row.items():
        if col.endswith("_mean") and val is not None:
            out[col] = float(val)
    return out


def build_campaign_timeline(
    experiments: list[dict[str, Any]],
    experiments_dir: Path,
    *,
    project_root: Path | None = None,
    sweep_kind: str | None = None,
    facet_filters: dict[str, Any] | None = None,
    title: str | None = None,
    apply_facet_filters: bool = False,
    plot_facet_note: str | None = None,
) -> CampaignTimeline:
    """
    Order runs by swept X and attach config deltas + per-run metrics.

    *experiments* are API records; on-disk ``config.yaml`` is loaded per run id.

    By default facet filters are **not** applied (full section run list for config
    timeline). Set ``apply_facet_filters=True`` to restrict steps (legacy CLI).
    """
    run_ids = run_ids_for_experiments(experiments, experiments_dir=experiments_dir)
    if not run_ids:
        raise ValueError("No completed runs with on-disk results")

    df = build_sweep_metrics_frame(run_ids, experiments_dir)
    if df.empty:
        raise ValueError("No metrics frame for campaign runs")

    kind = infer_sweep_kind(df, hint=sweep_kind)
    x_col = resolve_x_col(df, kind)
    x_label = {
        "noise_percent": "Label noise (%)",
        "under_train_per_class": "Under-train / class",
        "dataset_size": "Under-train / class",
    }.get(x_col, x_col.replace("_", " ").title())

    plot_df = filter_metrics_frame(df, facet_filters if apply_facet_filters else None)
    if plot_df.empty:
        raise ValueError("No runs match facet filters")

    if "run_id" not in plot_df.columns:
        raise ValueError("Metrics frame missing run_id")

    plot_df = plot_df.dropna(subset=[x_col]).sort_values(x_col)
    ordered_ids = [str(r) for r in plot_df["run_id"].tolist()]

    configs = [_tracked_flat(_load_run_config(rid, experiments_dir)) for rid in ordered_ids]
    shared = _shared_config(configs)

    id_to_name = {str(e.get("id") or ""): str(e.get("name") or e.get("id") or "")[:40] for e in experiments}
    id_to_row = {str(row["run_id"]): row.to_dict() for _, row in plot_df.iterrows()}

    steps: list[CampaignStep] = []
    for idx, run_id in enumerate(ordered_ids):
        changes: list[ConfigChange] = []
        if idx > 0:
            for key, old, new in find_tracked_differences(
                _load_run_config(ordered_ids[idx - 1], experiments_dir),
                _load_run_config(run_id, experiments_dir),
            ):
                changes.append(ConfigChange(key, _label_key(key), old, new))

        row = id_to_row.get(run_id, {})
        x_val = row.get(x_col)
        steps.append(
            CampaignStep(
                index=idx,
                run_id=run_id,
                run_name=id_to_name.get(run_id, run_id[:8]),
                x_value=None if x_val is None else float(x_val),
                x_label=x_label,
                changes_from_prev=tuple(changes),
                metrics=_metric_snapshot(row, x_col),
            )
        )

    facet_note = ""
    if facet_filters and apply_facet_filters:
        parts = [f"{k}={v}" for k, v in facet_filters.items() if v not in (None, "all")]
        if parts:
            facet_note = "Facet slice: " + ", ".join(parts)

    if plot_facet_note:
        facet_note = plot_facet_note

    sweep_label = kind.replace("_", " ")
    header = title or f"Campaign · {sweep_label} sweep · {len(steps)} runs"

    return CampaignTimeline(
        title=header,
        sweep_kind=kind,
        x_label=x_label,
        shared_config=shared,
        steps=tuple(steps),
        facet_note=facet_note,
    )


def _sweep_delta_summary(step: CampaignStep) -> str:
    if step.is_baseline:
        return "baseline"
    if not step.changes_from_prev:
        return "—"
    first = step.changes_from_prev[0].line()
    if len(step.changes_from_prev) > 1:
        return f"{first} (+{len(step.changes_from_prev) - 1})"
    return first


def _shared_config_lines(shared: dict[str, str]) -> list[str]:
    return [f"{k}: {v}" for k, v in shared.items()]


def _split_lines_two_columns(lines: list[str]) -> tuple[list[str], list[str]]:
    mid = (len(lines) + 1) // 2
    return lines[:mid], lines[mid:]


def _render_config_shared_page(
    timeline: CampaignTimeline,
    *,
    section_title: str | None = None,
    page_index: int = 0,
    page_count: int = 1,
) -> plt.Figure:
    """Page 1 — header + two-column shared setup."""
    fig, ax = plt.subplots(figsize=(8.5, 11.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.97
    if section_title:
        ax.text(0.04, y, section_title, fontsize=12, fontweight="bold", va="top")
        y -= 0.035
    ax.text(0.04, y, timeline.title, fontsize=10, fontweight="bold", va="top", color="#212121")
    y -= 0.028
    subtitle = timeline.facet_note or f"X axis: {timeline.x_label}"
    ax.text(0.04, y, subtitle, fontsize=8.5, va="top", color="#616161")
    y -= 0.04

    ax.text(0.04, y, "Shared setup (constant across runs)", fontsize=9, fontweight="bold", va="top")
    y -= 0.025

    lines = _shared_config_lines(timeline.shared_config)
    if not lines:
        ax.text(0.06, y, "(all tracked keys vary between runs)", fontsize=8, va="top", color="#757575")
    else:
        left, right = _split_lines_two_columns(lines)
        line_h = 0.022
        y_left = y
        for line in left:
            ax.text(0.04, y_left, line, fontsize=7.5, va="top", family="monospace")
            y_left -= line_h
        y_right = y
        for line in right:
            ax.text(0.52, y_right, line, fontsize=7.5, va="top", family="monospace")
            y_right -= line_h

    if page_count > 1:
        ax.text(0.98, 0.02, f"Config {page_index + 1}/{page_count}", fontsize=7, ha="right", color="#9e9e9e")
    fig.tight_layout()
    return fig


_SWEEPS_PER_TABLE_PAGE = 42


def _render_sweeps_table_page(
    timeline: CampaignTimeline,
    steps: tuple[CampaignStep, ...],
    *,
    page_index: int,
    page_count: int,
) -> plt.Figure:
    """Compact sweep table — one row per run, not one page per run."""
    fig, ax = plt.subplots(figsize=(8.5, 11.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.04,
        0.97,
        f"Sweep sequence · {timeline.x_label}",
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.04,
        0.94,
        "# · X · run · Δ vs previous",
        fontsize=7.5,
        va="top",
        color="#757575",
        family="monospace",
    )

    y = 0.91
    line_h = 0.021
    for step in steps:
        x_val = f"{step.x_value:g}" if step.x_value is not None else "—"
        run_short = step.run_name[:36]
        delta = _sweep_delta_summary(step)
        row = f"{step.index + 1:>2}  {x_val:>6}  {run_short:<36}  {delta}"
        ax.text(0.04, y, row, fontsize=7, va="top", family="monospace")
        y -= line_h

    ax.text(
        0.98,
        0.02,
        f"Sweeps {page_index + 1}/{page_count}",
        fontsize=7,
        ha="right",
        color="#9e9e9e",
    )
    fig.tight_layout()
    return fig


def append_config_timeline_pages(
    timeline: CampaignTimeline,
    pdf: PdfPages,
    *,
    section_title: str | None = None,
    **kwargs: Any,
) -> None:
    """Slim config export: page 1 = shared setup (2 columns), then sweep table pages."""
    del kwargs
    sweep_pages = (
        0
        if not timeline.steps
        else (len(timeline.steps) + _SWEEPS_PER_TABLE_PAGE - 1) // _SWEEPS_PER_TABLE_PAGE
    )
    total_pages = 1 + sweep_pages

    fig = _render_config_shared_page(
        timeline,
        section_title=section_title,
        page_index=0,
        page_count=total_pages,
    )
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)

    if not timeline.steps:
        return

    for page_idx in range(sweep_pages):
        start = page_idx * _SWEEPS_PER_TABLE_PAGE
        chunk = timeline.steps[start : start + _SWEEPS_PER_TABLE_PAGE]
        fig = _render_sweeps_table_page(
            timeline,
            chunk,
            page_index=page_idx,
            page_count=sweep_pages,
        )
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.35)
        plt.close(fig)


def build_campaign_timeline_figure(timeline: CampaignTimeline) -> plt.Figure:
    """Preview — shared config page."""
    return _render_config_shared_page(timeline)


def append_timeline_to_pdf(
    timeline: CampaignTimeline,
    pdf: PdfPages,
    *,
    section_title: str | None = None,
) -> None:
    append_config_timeline_pages(timeline, pdf, section_title=section_title)


def append_shared_config_page(
    timeline: CampaignTimeline,
    pdf: PdfPages,
    *,
    section_title: str | None = None,
) -> None:
    append_config_timeline_pages(timeline, pdf, section_title=section_title)


def _render_group_diff_figure(
    *,
    group_label: str,
    changes: tuple[ConfigChange, ...],
    previous_label: str,
) -> plt.Figure:
    """How this campaign group differs from the previous exported group."""
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.92, group_label, fontsize=11, fontweight="bold", va="top")
    ax.text(
        0.04,
        0.84,
        f"Δ shared setup vs previous campaign ({previous_label}):",
        fontsize=9,
        fontweight="bold",
        va="top",
        color="#37474f",
    )
    y = 0.76
    if not changes:
        ax.text(
            0.06,
            y,
            "Same tracked shared parameters as previous campaign.",
            fontsize=9,
            va="top",
            color="#616161",
        )
    else:
        for change in changes:
            ax.text(0.06, y, f"• {change.line()}", fontsize=9, va="top", family="monospace")
            y -= 0.055

    fig.tight_layout()
    return fig


def _shared_config_changes(
    previous: dict[str, str],
    current: dict[str, str],
) -> tuple[ConfigChange, ...]:
    changes: list[ConfigChange] = []
    keys = sorted(set(previous) | set(current))
    for key in keys:
        old = previous.get(key)
        new = current.get(key)
        if old == new:
            continue
        changes.append(ConfigChange(key, key, old, new))
    return tuple(changes)


def append_group_diff_page(
    pdf: PdfPages,
    *,
    group_label: str,
    current_timeline: CampaignTimeline,
    previous_timeline: CampaignTimeline,
    previous_label: str,
) -> None:
    changes = _shared_config_changes(previous_timeline.shared_config, current_timeline.shared_config)
    fig = _render_group_diff_figure(
        group_label=group_label,
        changes=changes,
        previous_label=previous_label,
    )
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def config_timeline_entries(
    timeline: CampaignTimeline,
    *,
    section_title: str | None = None,
) -> list[tuple[str, bool]]:
    """Flat (text, is_heading) lines for paginated config-only rendering."""
    entries: list[tuple[str, bool]] = []
    if section_title:
        entries.append((section_title, True))
    entries.append((timeline.title, True))
    subtitle = timeline.facet_note or f"X axis: {timeline.x_label}"
    entries.append((subtitle, False))
    entries.append(("", False))
    entries.append(("Shared setup (constant across runs)", True))
    if timeline.shared_config:
        for key, value in timeline.shared_config.items():
            entries.append((f"  {key}: {value}", False))
    else:
        entries.append(("  (all tracked keys vary between runs)", False))
    entries.append(("", False))

    for step in timeline.steps:
        x_disp = (
            f"{step.x_label} = {step.x_value:g}"
            if step.x_value is not None
            else step.x_label
        )
        entries.append((f"Sweep {step.index + 1} · {x_disp} · {step.run_name}", True))
        if step.is_baseline:
            entries.append(("  baseline sweep — no prior run", False))
        elif step.changes_from_prev:
            entries.append(("  Δ vs previous sweep:", False))
            for change in step.changes_from_prev:
                entries.append((f"    {change.line()}", False))
        else:
            entries.append(("  (no config delta detected)", False))
        entries.append(("", False))
    return entries


_PAGE_TOP = 0.95
_PAGE_BOTTOM = 0.05
_LINE_HEIGHT = 0.028
_HEADING_HEIGHT = 0.034
_LEFT = 0.04
_FONTSIZE = 8.5
_HEADING_FONTSIZE = 9.5


def _lines_per_page() -> int:
    usable = _PAGE_TOP - _PAGE_BOTTOM
    return max(20, int(usable / _LINE_HEIGHT) - 2)


def _render_timeline_page(
    entries: list[tuple[str, bool]],
    *,
    page_index: int,
    page_count: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 11.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = _PAGE_TOP
    for text, is_heading in entries:
        if not text:
            y -= _LINE_HEIGHT * 0.6
            continue
        step = _HEADING_HEIGHT if is_heading else _LINE_HEIGHT
        if y - step < _PAGE_BOTTOM:
            break
        ax.text(
            _LEFT,
            y,
            text,
            transform=ax.transAxes,
            fontsize=_HEADING_FONTSIZE if is_heading else _FONTSIZE,
            fontweight="bold" if is_heading else "normal",
            va="top",
            family="monospace",
        )
        y -= step
    if page_count > 1:
        ax.text(
            0.98,
            0.02,
            f"Config timeline {page_index + 1}/{page_count}",
            transform=ax.transAxes,
            fontsize=7,
            color="#757575",
            ha="right",
            va="bottom",
        )
    fig.tight_layout()
    return fig


def _paginate_entries(entries: list[tuple[str, bool]]) -> list[list[tuple[str, bool]]]:
    per_page = _lines_per_page()
    pages: list[list[tuple[str, bool]]] = []
    current: list[tuple[str, bool]] = []
    used = 0
    for text, is_heading in entries:
        cost = 1 if text else 0.4
        if current and used + cost > per_page:
            pages.append(current)
            current = []
            used = 0
        current.append((text, is_heading))
        used += cost
    if current:
        pages.append(current)
    return pages or [[]]


def save_timeline_figure(fig: plt.Figure, path: Path, *, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.2)
    else:
        fmt = "png" if suffix in {".png", ""} else suffix.lstrip(".")
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    return path


def timeline_figure_to_pdf_bytes(fig: plt.Figure) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0.2)
    return buf.getvalue()

