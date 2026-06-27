"""
Thesis-style schematic figures from experiment config (+ optional dataset splits).

Panel A: how train / eval pools are defined (set logic + counts).
Panel B: uncertainty signal computation pipeline (primitives → signals → metrics).
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from uqlab.data.dataset_registry import get_dataset_spec
from uqlab.shared.config.classification import ExperimentConfig
from uqlab.runner.phases.config_view import (
    RunConfigView,
    extract_run_config,
)
from uqlab.evaluation.reporting.sweep_plot_pools import (
    PoolExpectations,
    pool_expectations_from_data_config,
)
from uqlab.evaluation.signals.formulas import SignalFormulaSpec, experiment_signal_formula_specs
from uqlab.shared.config.signals import flatten_signals, normalize_evaluation_signals, prune_signals_for_runtime

_POOL_STYLES = {
    "train": {"face": "#ECEFF1", "edge": "#455A64", "title": r"Train $\mathcal{T}$"},
    "clean": {"face": "#FAFAFA", "edge": "#757575", "title": r"Clean $\mathcal{E}_{\mathrm{clean}}$"},
    "aleatoric": {"face": "#D6EAF8", "edge": "#2471A3", "title": r"Aleatoric $\mathcal{E}_{\mathrm{alea}}$"},
    "epistemic": {"face": "#D5F5E3", "edge": "#229954", "title": r"Epistemic $\mathcal{E}_{\mathrm{epi}}$"},
}


@dataclass(frozen=True)
class ThesisDiagramInputs:
    """Everything needed to render both panels."""

    run_view: RunConfigView
    pool_expectations: PoolExpectations
    split_counts: dict[str, int] | None
    enabled_signals: list[str]
    signal_specs: dict[str, SignalFormulaSpec]
    empirical: bool
    dataset_name: str
    num_classes: int
    seed: int


def _estimate_train_count(view: RunConfigView, num_classes: int) -> int:
    under = set(view.under_supported_classes)
    n_under = len(under) * int(view.under_train_per_class)
    n_regular = (num_classes - len(under)) * int(view.regular_train_per_class)
    return n_under + n_regular


def experiment_config_from_yaml_dict(cfg_dict: dict[str, Any]) -> ExperimentConfig:
    """Parse nested run YAML dict into ``ExperimentConfig``."""
    import tempfile
    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(cfg_dict, handle, sort_keys=False)
        path = Path(handle.name)
    try:
        return ExperimentConfig.from_yaml(path)
    finally:
        path.unlink(missing_ok=True)


def experiment_config_from_workflow(workflow: dict[str, Any]) -> ExperimentConfig:
    """Build ``ExperimentConfig`` from progressive UI workflow session state."""
    from uqlab_orchestrator.run_spec import build_run_yaml

    return experiment_config_from_yaml_dict(build_run_yaml(workflow))


def load_thesis_diagram_inputs(
    config: ExperimentConfig,
    project_root: Path,
    *,
    seed: int,
    empirical: bool = True,
) -> ThesisDiagramInputs:
    """
    Build diagram inputs from YAML config.

    When *empirical* is True, loads the dataset and samples splits for real counts.
    Otherwise only symbolic expectations and estimated train size are shown.
    """
    view = extract_run_config(config)
    data = config.data
    assert data is not None and config.model is not None and config.evaluation is not None

    dataset_name = view.dataset_name
    num_classes = get_dataset_spec(dataset_name).num_classes
    pool_expectations = pool_expectations_from_data_config(
        {
            "under_supported_classes": data.under_supported_classes,
            "under_train_per_class": data.under_train_per_class,
            "regular_train_per_class": data.regular_train_per_class,
            "aleatoric_noise_percentage": data.aleatoric_noise_percentage,
        },
        seed=seed,
    )

    split_counts: dict[str, int] | None = None
    if empirical:
        from uqlab.data.setup import prepare_experiment_data

        data_ctx = prepare_experiment_data(config, project_root, seed=seed)
        spec = data_ctx.split_spec
        split_counts = {
            "train": len(spec.train_indices),
            "clean": len(spec.clean_eval_indices),
            "aleatoric": len(spec.aleatoric_eval_indices),
            "epistemic": len(spec.epistemic_eval_indices),
        }
    else:
        split_counts = {
            "train": _estimate_train_count(view, num_classes),
            "clean": int(view.eval_per_group),
            "aleatoric": int(view.eval_per_group) if pool_expectations.aleatoric_pool_expected else 0,
            "epistemic": int(view.eval_per_group) if pool_expectations.epistemic_pool_expected else 0,
        }

    pruned = prune_signals_for_runtime(
        normalize_evaluation_signals(config.evaluation.signals),
        mc_passes=view.mc_passes,
        dropout=float(view.dropout),
    )
    enabled = flatten_signals(pruned)
    all_specs = experiment_signal_formula_specs(
        top_k=view.top_k,
        mc_passes=view.mc_passes,
    )
    signal_specs = {name: all_specs[name] for name in enabled if name in all_specs}

    return ThesisDiagramInputs(
        run_view=view,
        pool_expectations=pool_expectations,
        split_counts=split_counts,
        enabled_signals=enabled,
        signal_specs=signal_specs,
        empirical=empirical,
        dataset_name=dataset_name,
        num_classes=num_classes,
        seed=seed,
    )


def _panel_heading(ax, title: str, subtitle: str | None = None) -> None:
    """Panel title above content — avoids overlap with the first row of boxes."""
    ax.text(
        0.0,
        1.02,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
    )
    if subtitle:
        ax.text(
            0.0,
            0.985,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#616161",
            clip_on=False,
        )


def _draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    lines: list[str],
    face: str,
    edge: str,
    dashed: bool = False,
    title_math: bool = True,
) -> None:
    linestyle = (0, (4, 4)) if dashed else "-"
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(patch)
    title_y = y + h - 0.025
    body_y = y + h - 0.065
    if title_math and "$" in title:
        ax.text(
            x + w / 2,
            title_y,
            title,
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            transform=ax.transAxes,
            zorder=2,
        )
    else:
        ax.text(
            x + w / 2,
            title_y,
            title,
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            transform=ax.transAxes,
            zorder=2,
        )
    body = "\n".join(line for line in lines if line is not None)
    if body:
        ax.text(
            x + 0.025,
            body_y,
            body,
            ha="left",
            va="top",
            fontsize=7.5,
            transform=ax.transAxes,
            linespacing=1.25,
            zorder=2,
        )


def _draw_arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        color="#424242",
        transform=ax.transAxes,
        zorder=3,
    )
    ax.add_patch(arrow)


def render_split_panel(ax, inputs: ThesisDiagramInputs) -> None:
    """Panel A: config parameters and eval-pool set logic."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_clip_on(False)

    mode = "empirical counts" if inputs.empirical else "symbolic counts"
    _panel_heading(ax, "A. Data partitioning", subtitle=mode)

    view = inputs.run_view
    under_str = ", ".join(str(c) for c in view.under_supported_classes)
    cfg_lines = [
        f"Dataset: {inputs.dataset_name} ({inputs.num_classes} classes)",
        f"Under classes: {{{under_str}}}",
        f"Under-train / class: {view.under_train_per_class}",
        f"Regular-train / class: {view.regular_train_per_class}",
        f"Label noise: {view.aleatoric_noise_percentage}%",
        f"Eval / group: {view.eval_per_group} · seed {inputs.seed}",
    ]
    _draw_box(
        ax, 0.0, 0.54, 0.46, 0.38,
        title="Configuration",
        lines=cfg_lines,
        face="#FFFDE7",
        edge="#F9A825",
        title_math=False,
    )

    logic_lines = [
        r"$M_{\mathrm{under}}(x)$: under-supported class",
        r"$M_{\mathrm{noise}}(x)$: noisy label",
        r"$\mathcal{T}$: train indices",
        r"($M_u \equiv M_{\mathrm{under}}$, $M_n \equiv M_{\mathrm{noise}}$)",
        "",
        r"$\mathcal{E}_{\mathrm{clean}} = \neg M_u \wedge \neg M_n \wedge \neg \mathcal{T}$",
        r"$\mathcal{E}_{\mathrm{alea}} = \neg M_u \wedge M_n \wedge \neg \mathcal{T}$",
        r"$\mathcal{E}_{\mathrm{epi}} = M_u \wedge \neg M_n \wedge \neg \mathcal{T}$",
    ]
    _draw_box(
        ax, 0.52, 0.54, 0.48, 0.38,
        title="Set definitions",
        lines=logic_lines,
        face="#F3E5F5",
        edge="#7B1FA2",
        title_math=False,
    )

    counts = inputs.split_counts or {}
    exp = inputs.pool_expectations
    pool_defs = [
        (
            "train",
            [
                r"$n_c =$ under / regular budget",
                rf"$|\mathcal{{T}}| = {counts.get('train', '?')}$",
            ],
            False,
        ),
        (
            "clean",
            [
                "Regular · clean",
                rf"$|E| = {counts.get('clean', '?')}$",
            ],
            False,
        ),
        (
            "aleatoric",
            [
                "Regular · noisy",
                rf"$|E| = {counts.get('aleatoric', '?')}$",
                "expected" if exp.aleatoric_pool_expected else "not expected",
            ],
            not exp.aleatoric_pool_expected,
        ),
        (
            "epistemic",
            [
                "Under · clean",
                rf"$|E| = {counts.get('epistemic', '?')}$",
                "expected" if exp.epistemic_pool_expected else "not expected",
            ],
            not exp.epistemic_pool_expected,
        ),
    ]

    xs = [0.0, 0.255, 0.51, 0.765]
    for (key, lines, dashed), x in zip(pool_defs, xs):
        style = _POOL_STYLES[key]
        _draw_box(
            ax, x, 0.0, 0.235, 0.46,
            title=style["title"],
            lines=lines,
            face=style["face"],
            edge=style["edge"],
            dashed=dashed,
        )


def _draw_signal_table(ax, inputs: ThesisDiagramInputs) -> None:
    shown = inputs.enabled_signals[:8]
    if not shown:
        ax.text(0.04, 0.45, "No enabled signals in config.", fontsize=8, transform=ax.transAxes)
        return

    rows = []
    for name in shown:
        spec = inputs.signal_specs.get(name)
        rows.append([
            spec.label if spec else name.replace("_", " "),
            spec.formula if spec else "—",
        ])

    n_rows = len(rows)
    table_h = min(0.52, 0.08 + 0.055 * n_rows)
    table = ax.table(
        cellText=rows,
        colLabels=["Signal", "Formula"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.20, 1.0, table_h],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.8)
    table.scale(1.0, 1.25)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CFD8DC")
        if row == 0:
            cell.set_facecolor("#ECEFF1")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#FAFAFA" if row % 2 else "#FFFFFF")
        if col == 0:
            cell.set_width(0.22)
        elif col == 1:
            cell.set_width(0.78)
        cell.PAD = 0.03

    if len(inputs.enabled_signals) > len(shown):
        ax.text(
            0.0,
            0.17,
            f"+{len(inputs.enabled_signals) - len(shown)} more enabled in config",
            fontsize=7,
            color="#757575",
            transform=ax.transAxes,
        )


def render_signal_panel(ax, inputs: ThesisDiagramInputs) -> None:
    """Panel B: pipeline → signal table → evaluation metrics row."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_clip_on(False)

    view = inputs.run_view
    _panel_heading(
        ax,
        "B. Uncertainty signal computation",
        subtitle=f"{view.architecture} · MC={view.mc_passes} · top-k={view.top_k}",
    )

    stages = [
        (0.00, "Input", ["eval batch", r"train $\mathcal{T}$"]),
        (0.19, "Model", [view.architecture[:12], f"dropout={view.dropout}"]),
        (0.38, "Primitives", ["logits", "DualXDA", f"MC K={view.mc_passes}"]),
        (0.57, r"Signals $s_j$", ["one value", "per sample"]),
        (0.76, "Aggregate", [r"pool mean", "AUROC"]),
    ]
    box_y, box_h = 0.74, 0.14
    for x, title, lines in stages:
        _draw_box(
            ax, x, box_y, 0.17, box_h,
            title=title,
            lines=lines,
            face="#E3F2FD",
            edge="#1565C0",
            title_math=False,
        )
    for x0, x1 in zip([0.17, 0.36, 0.55, 0.74], [0.19, 0.38, 0.57, 0.76]):
        _draw_arrow(ax, x0, box_y + box_h / 2, x1, box_y + box_h / 2)

    ax.text(
        0.0,
        0.70,
        "Enabled signals",
        fontsize=9,
        fontweight="bold",
        transform=ax.transAxes,
        va="bottom",
    )
    _draw_signal_table(ax, inputs)

    metrics = [
        (
            "Pool mean",
            [r"$\bar{s}_G = \mathrm{mean}_{j \in \mathcal{E}_G}\, s_j$"],
        ),
        (
            "Aleatoric AUROC",
            [r"rank $s$ on $\mathcal{E}_{\mathrm{alea}}$", "vs noisy labels"],
        ),
        (
            "Epistemic AUROC",
            [r"rank $s$ on $\mathcal{E}_{\mathrm{epi}}$", "vs under-train"],
        ),
    ]
    for idx, (title, lines) in enumerate(metrics):
        _draw_box(
            ax,
            idx * 0.335 + 0.01,
            0.0,
            0.31,
            0.16,
            title=title,
            lines=lines,
            face="#FFF3E0",
            edge="#EF6C00",
            title_math=False,
        )


def build_thesis_figure(
    inputs: ThesisDiagramInputs,
    *,
    panels: tuple[str, ...] = ("a", "b"),
) -> plt.Figure:
    """Thesis schematic (matplotlib Figure). ``panels``: ``a`` = config/pools, ``b`` = signal pipeline."""
    normalized = tuple(p.lower() for p in panels) or ("a", "b")
    show_a = "a" in normalized
    show_b = "b" in normalized
    if not show_a and not show_b:
        show_a, show_b = True, True

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "mathtext.fontset": "dejavusans",
        }
    )
    if show_a and show_b:
        fig = plt.figure(figsize=(11.0, 9.5), dpi=100)
        fig.suptitle(
            "Uncertainty quantification experiment schematic",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], top=0.93, bottom=0.04, hspace=0.42)
        ax_a = fig.add_subplot(gs[0])
        ax_b = fig.add_subplot(gs[1])
        render_split_panel(ax_a, inputs)
        render_signal_panel(ax_b, inputs)
    elif show_a:
        fig = plt.figure(figsize=(11.0, 5.2), dpi=100)
        fig.suptitle(
            "Experiment config — train / eval pools",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        ax_a = fig.add_subplot(111)
        render_split_panel(ax_a, inputs)
    else:
        fig = plt.figure(figsize=(11.0, 6.0), dpi=100)
        fig.suptitle(
            "Results pipeline — signals and metrics",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        ax_b = fig.add_subplot(111)
        render_signal_panel(ax_b, inputs)
    return fig


def save_thesis_figure(fig: plt.Figure, path: Path, *, dpi: int = 300) -> Path:
    """Save figure to PDF/PNG/SVG based on suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    elif suffix in {".png", ".jpg", ".jpeg"}:
        fig.savefig(path, format="png" if suffix == ".png" else "jpeg", dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    elif suffix == ".svg":
        fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.15)
    else:
        out = path.with_suffix(".pdf")
        fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.15)
        path = out
    return path


def thesis_figure_to_bytes(fig: plt.Figure, fmt: str = "pdf", *, dpi: int = 300) -> bytes:
    """Serialize figure to PDF or PNG bytes (for Streamlit download buttons)."""
    buffer = BytesIO()
    fmt = fmt.lower().lstrip(".")
    if fmt == "pdf":
        fig.savefig(buffer, format="pdf", bbox_inches="tight", pad_inches=0.15)
    elif fmt in {"png", "jpg", "jpeg"}:
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    elif fmt == "svg":
        fig.savefig(buffer, format="svg", bbox_inches="tight", pad_inches=0.15)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}")
    return buffer.getvalue()
