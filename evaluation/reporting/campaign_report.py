"""
Campaign PDF report — config timeline + sweep line plot(s).
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from uqlab.evaluation.reporting.campaign_config_timeline import (
    CampaignTimeline,
    append_config_timeline_pages,
    append_group_diff_page,
    build_campaign_timeline,
)
from uqlab.evaluation.reporting.campaign_sections import (
    CampaignSection,
    split_campaign_sections,
)
from uqlab.evaluation.reporting.sweep_line_plot import (
    build_sweep_line_plot,
    build_sweep_metrics_frame,
    list_plottable_signals,
    resolve_sweep_trace_xy,
)

PdfLayout = Literal["by_section", "by_metric"]


@dataclass(frozen=True)
class CampaignExportBundle:
    """One smart-group campaign for PDF export."""

    label: str
    experiments: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CampaignReportSummary:
    """Metadata for a multi-section campaign PDF."""

    title: str
    sections: tuple[CampaignTimeline, ...]
    section_labels: tuple[str, ...]
    group_labels: tuple[str, ...] = ()
    layout: PdfLayout = "by_section"

    @property
    def n_runs(self) -> int:
        return sum(t.n_runs for t in self.sections)

    @property
    def sweep_kind(self) -> str:
        return self.sections[0].sweep_kind if self.sections else "campaign"


@dataclass(frozen=True)
class _PlotSlice:
    label: str
    section: CampaignSection
    timeline: CampaignTimeline
    run_ids: tuple[str, ...]
    available_signals: tuple[str, ...]


def _plot_traces_on_axes(
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    plot_data: dict[str, Any],
) -> None:
    traces = plot_data.get("traces") or []
    for trace in traces:
        axis = ax_right if trace.get("yaxis") == "right" else ax_left
        dash = trace.get("dash")
        if dash == "dot":
            linestyle = ":"
        elif dash in ("dash", "dashed"):
            linestyle = "--"
        else:
            linestyle = "-"
        xs, ys = resolve_sweep_trace_xy(traces, trace)
        if not xs:
            continue
        axis.plot(
            xs,
            ys,
            marker="o",
            linestyle=linestyle,
            linewidth=1.8,
            markersize=4,
            color=trace.get("color", "#333333"),
            label=trace.get("name", "series"),
        )
    ax_left.set_xlabel(plot_data.get("x_label", "X"), fontsize=8)
    ax_left.set_ylabel(plot_data.get("y_left_title", "Signal"), fontsize=7)
    ax_right.set_ylabel(plot_data.get("y_right_title", "Accuracy"), fontsize=7)
    ax_right.set_ylim(0, 1)
    ax_left.tick_params(labelsize=7)
    ax_right.tick_params(labelsize=7)
    handles, labels = [], []
    for axis in (ax_left, ax_right):
        h, lab = axis.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(lab)
    if handles:
        ax_left.legend(handles, labels, fontsize=6, loc="best", frameon=False)


def _sweep_plot_figure(
    plot_data: dict[str, Any],
    *,
    title: str | None = None,
) -> plt.Figure:
    """Single sweep plot — no redundant caption footer."""
    fig, ax_left = plt.subplots(figsize=(8.5, 4.0))
    ax_right = ax_left.twinx()
    _plot_traces_on_axes(ax_left, ax_right, plot_data)
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)
    fig.tight_layout()
    return fig


def _metric_grouped_figure(
    signal: str,
    signal_label: str,
    slices: list[_PlotSlice],
    experiments_dir: Path,
    *,
    facet_filters: dict[str, Any] | None,
) -> plt.Figure | None:
    """One page per metric — subplot per sweep section/campaign slice."""
    panels: list[tuple[_PlotSlice, dict[str, Any]]] = []
    for sl in slices:
        if signal not in sl.available_signals:
            continue
        try:
            payload = build_sweep_line_plot(
                list(sl.run_ids),
                experiments_dir,
                signal=signal,
                facet_filters=facet_filters,
                sweep_kind=sl.timeline.sweep_kind,
            )
        except (ValueError, TypeError):
            continue
        panels.append((sl, payload.to_dict()))

    if not panels:
        return None

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(8.5, 3.6 * min(n, 3)), squeeze=False)
    for col, (sl, plot_data) in enumerate(panels):
        ax_left = axes[0, col]
        ax_right = ax_left.twinx()
        _plot_traces_on_axes(ax_left, ax_right, plot_data)
        ax_left.set_title(sl.label, fontsize=8, pad=6)

    fig.suptitle(signal_label, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def _facet_note(facet_filters: dict[str, Any] | None) -> str:
    if not facet_filters:
        return ""
    parts = [f"{k}={v}" for k, v in facet_filters.items() if v not in (None, "all")]
    if not parts:
        return ""
    return "Plot facet slice: " + ", ".join(parts)


def _signals_to_plot(
    available: list[str],
    *,
    primary: str | None,
    include_all_signals: bool,
    extra_signals: int,
) -> list[str]:
    if include_all_signals:
        return list(available)
    if primary and primary in available:
        ordered = [primary]
    elif available:
        ordered = [available[0]]
    else:
        return []
    if extra_signals <= 0:
        return ordered
    for alt in available:
        if alt in ordered:
            continue
        ordered.append(alt)
        if len(ordered) >= 1 + extra_signals:
            break
    return ordered


def _build_plot_slices(
    experiments: list[dict[str, Any]],
    experiments_dir: Path,
    *,
    project_root: Path | None,
    sweep_kind: str | None,
    facet_filters: dict[str, Any] | None,
    title: str | None,
    plot_note: str,
) -> list[_PlotSlice]:
    slices: list[_PlotSlice] = []
    sections = split_campaign_sections(experiments, experiments_dir=experiments_dir)
    for section in sections:
        section_kind = section.sweep_kind or sweep_kind
        section_title = title or f"Campaign · {len(section.experiments)} runs"
        timeline = build_campaign_timeline(
            list(section.experiments),
            experiments_dir,
            project_root=project_root,
            sweep_kind=section_kind,
            facet_filters=facet_filters,
            title=section_title,
            apply_facet_filters=False,
            plot_facet_note=plot_note or None,
        )
        run_ids = tuple(step.run_id for step in timeline.steps)
        if len(run_ids) < 2:
            continue
        df = build_sweep_metrics_frame(list(run_ids), experiments_dir)
        available = tuple(list_plottable_signals(df, timeline.sweep_kind))
        if not available:
            continue
        slices.append(
            _PlotSlice(
                label=section.label,
                section=section,
                timeline=timeline,
                run_ids=run_ids,
                available_signals=available,
            )
        )
    return slices


def _append_by_section_plots(
    pdf: PdfPages,
    slices: list[_PlotSlice],
    experiments_dir: Path,
    *,
    facet_filters: dict[str, Any] | None,
    signal: str | None,
    include_all_signals: bool,
    extra_signals: int,
) -> None:
    for sl in slices:
        available = list(sl.available_signals)
        primary = signal if signal in available else None
        for sig in _signals_to_plot(
            available,
            primary=primary,
            include_all_signals=include_all_signals,
            extra_signals=extra_signals,
        ):
            try:
                payload = build_sweep_line_plot(
                    list(sl.run_ids),
                    experiments_dir,
                    signal=sig,
                    facet_filters=facet_filters,
                    sweep_kind=sl.timeline.sweep_kind,
                )
            except (ValueError, TypeError):
                continue
            fig = _sweep_plot_figure(
                payload.to_dict(),
                title=f"{sl.label} · {payload.signal_label}",
            )
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)
            plt.close(fig)


def _append_by_metric_plots(
    pdf: PdfPages,
    slices: list[_PlotSlice],
    experiments_dir: Path,
    *,
    facet_filters: dict[str, Any] | None,
    signal: str | None,
    include_all_signals: bool,
    extra_signals: int,
) -> None:
    if not slices:
        return
    all_available: list[str] = []
    seen: set[str] = set()
    for sl in slices:
        for s in sl.available_signals:
            if s not in seen:
                seen.add(s)
                all_available.append(s)

    to_plot = _signals_to_plot(
        all_available,
        primary=signal if signal in all_available else None,
        include_all_signals=include_all_signals,
        extra_signals=extra_signals,
    )
    for sig in to_plot:
        from uqlab.shared.notebook_utils.signals import SIGNAL_LABELS

        label = SIGNAL_LABELS.get(sig, sig)
        fig = _metric_grouped_figure(
            sig,
            label,
            slices,
            experiments_dir,
            facet_filters=facet_filters,
        )
        if fig is None:
            continue
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)


def _append_campaign_to_pdf(
    pdf: PdfPages,
    experiments: list[dict[str, Any]],
    experiments_dir: Path,
    *,
    project_root: Path | None,
    sweep_kind: str | None,
    facet_filters: dict[str, Any] | None,
    signal: str | None,
    title: str | None,
    extra_signals: int,
    include_all_signals: bool,
    layout: PdfLayout,
    group_label: str | None = None,
    previous_first_timeline: CampaignTimeline | None = None,
) -> tuple[list[CampaignTimeline], list[str]]:
    plot_note = _facet_note(facet_filters)
    slices = _build_plot_slices(
        experiments,
        experiments_dir,
        project_root=project_root,
        sweep_kind=sweep_kind,
        facet_filters=facet_filters,
        title=title,
        plot_note=plot_note,
    )
    if not slices:
        raise ValueError("Need at least 2 plottable completed runs for campaign PDF")

    timelines = [sl.timeline for sl in slices]
    labels = [sl.label for sl in slices]

    for section_idx, sl in enumerate(slices):
        if section_idx == 0 and previous_first_timeline is not None and group_label:
            append_group_diff_page(
                pdf,
                group_label=group_label,
                current_timeline=sl.timeline,
                previous_timeline=previous_first_timeline,
                previous_label=previous_first_timeline.title,
            )
        append_config_timeline_pages(
            sl.timeline,
            pdf,
            section_title=sl.label,
        )

    if layout == "by_metric":
        _append_by_metric_plots(
            pdf,
            slices,
            experiments_dir,
            facet_filters=facet_filters,
            signal=signal,
            include_all_signals=include_all_signals,
            extra_signals=extra_signals,
        )
    else:
        _append_by_section_plots(
            pdf,
            slices,
            experiments_dir,
            facet_filters=facet_filters,
            signal=signal,
            include_all_signals=include_all_signals,
            extra_signals=extra_signals,
        )

    return timelines, labels


def build_campaign_report_pdf(
    experiments: list[dict[str, Any]],
    experiments_dir: Path,
    output: Path | None = None,
    *,
    project_root: Path | None = None,
    sweep_kind: str | None = None,
    facet_filters: dict[str, Any] | None = None,
    signal: str | None = None,
    title: str | None = None,
    extra_signals: int = 0,
    include_all_signals: bool = True,
    layout: PdfLayout = "by_section",
) -> tuple[bytes, CampaignReportSummary]:
    bundles = [
        CampaignExportBundle(
            label=title or "Campaign",
            experiments=tuple(experiments),
        )
    ]
    return build_multi_campaign_report_pdf(
        bundles,
        experiments_dir,
        output=output,
        project_root=project_root,
        sweep_kind=sweep_kind,
        facet_filters=facet_filters,
        signal=signal,
        title=title,
        extra_signals=extra_signals,
        include_all_signals=include_all_signals,
        layout=layout,
    )


def build_multi_campaign_report_pdf(
    bundles: list[CampaignExportBundle],
    experiments_dir: Path,
    output: Path | None = None,
    *,
    project_root: Path | None = None,
    sweep_kind: str | None = None,
    facet_filters: dict[str, Any] | None = None,
    signal: str | None = None,
    title: str | None = None,
    extra_signals: int = 0,
    include_all_signals: bool = True,
    layout: PdfLayout = "by_section",
) -> tuple[bytes, CampaignReportSummary]:
    if not bundles:
        raise ValueError("No campaigns to export")

    all_timelines: list[CampaignTimeline] = []
    all_section_labels: list[str] = []
    group_labels: list[str] = []
    previous_first_timeline: CampaignTimeline | None = None

    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        for bundle in bundles:
            completed = [e for e in bundle.experiments if e.get("status") == "completed"]
            timelines, section_labels = _append_campaign_to_pdf(
                pdf,
                completed,
                experiments_dir,
                project_root=project_root,
                sweep_kind=sweep_kind,
                facet_filters=facet_filters,
                signal=signal,
                title=title or bundle.label,
                extra_signals=extra_signals,
                include_all_signals=include_all_signals,
                layout=layout,
                group_label=bundle.label if len(bundles) > 1 else None,
                previous_first_timeline=previous_first_timeline,
            )
            all_timelines.extend(timelines)
            all_section_labels.extend(section_labels)
            group_labels.append(bundle.label)
            if timelines:
                previous_first_timeline = timelines[0]

    n_groups = len(bundles)
    n_runs = sum(t.n_runs for t in all_timelines)
    layout_tag = "metric" if layout == "by_metric" else "section"
    summary_title = title or (
        f"Campaign report ({layout_tag}) · {n_groups} groups · {n_runs} runs"
        if n_groups > 1
        else f"Campaign report ({layout_tag}) · {n_runs} runs"
    )
    summary = CampaignReportSummary(
        title=summary_title,
        sections=tuple(all_timelines),
        section_labels=tuple(all_section_labels),
        group_labels=tuple(group_labels),
        layout=layout,
    )
    pdf_bytes = buffer.getvalue()
    if output is not None:
        Path(output).write_bytes(pdf_bytes)
    return pdf_bytes, summary


def campaign_report_filename(
    summary: CampaignReportSummary | CampaignTimeline,
    *,
    prefix: str = "campaign_report",
) -> str:
    if isinstance(summary, CampaignReportSummary):
        n_runs = summary.n_runs
        n_sections = len(summary.sections)
        n_groups = len(summary.group_labels) or 1
        kind = summary.sweep_kind.replace("_", "-")
        layout = summary.layout
    else:
        n_runs = summary.n_runs
        n_sections = 1
        n_groups = 1
        kind = summary.sweep_kind.replace("_", "-")
        layout = "by_section"
    group_tag = f"{n_groups}grp-" if n_groups > 1 else ""
    section_tag = f"{n_sections}sec-" if n_sections > 1 else ""
    layout_tag = "by-metric_" if layout == "by_metric" else ""
    return f"{prefix}_{layout_tag}{group_tag}{section_tag}{kind}_{n_runs}runs.pdf"
