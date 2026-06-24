"""Streamlit UI for thesis-style experiment schematics."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

from uqlab.shared.config.classification import ExperimentConfig
from uqlab.evaluation.pipeline.thesis_diagram import (
    build_thesis_figure,
    experiment_config_from_workflow,
    load_thesis_diagram_inputs,
    thesis_figure_to_bytes,
)
from uqlab.runtime_paths import experiments_root, resolve_experiment_results_dir


def _resolve_run_config_path(
    experiments: list[dict[str, Any]] | None,
    *,
    experiments_dir: Path | None = None,
) -> Path | None:
    if not experiments:
        return None
    exp_dir = experiments_dir or experiments_root()
    for exp in experiments:
        eid = str(exp.get("id") or "")
        if not eid:
            continue
        results_dir = resolve_experiment_results_dir(
            eid,
            results_path=exp.get("results_path"),
        )
        cfg_path = results_dir.parent / "config.yaml"
        if cfg_path.is_file():
            return cfg_path
    return None


def render_thesis_diagram_panel(
    *,
    config: ExperimentConfig | None = None,
    workflow: dict[str, Any] | None = None,
    config_path: Path | None = None,
    experiments: list[dict[str, Any]] | None = None,
    project_root: Path,
    key_prefix: str = "thesis_diagram",
    default_symbolic: bool = True,
) -> None:
    """Compact thesis schematic: generate once, preview + PDF/PNG download."""
    resolved_config = config
    if resolved_config is None and config_path is not None:
        resolved_config = ExperimentConfig.from_yaml(config_path)
    elif resolved_config is None and workflow is not None:
        resolved_config = experiment_config_from_workflow(workflow)
    elif resolved_config is None and experiments:
        run_cfg = _resolve_run_config_path(experiments)
        if run_cfg is not None:
            resolved_config = ExperimentConfig.from_yaml(run_cfg)

    if resolved_config is None:
        st.caption("Thesis schematic unavailable (no config on disk).")
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        symbolic = st.checkbox(
            "Symbolic counts (fast)",
            value=default_symbolic,
            key=f"{key_prefix}_symbolic",
        )
    with c2:
        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=10_000_000,
            value=int(resolved_config.seed),
            step=1,
            key=f"{key_prefix}_seed",
        )

    cache_key = f"{key_prefix}_figure_inputs"
    params_key = f"{key_prefix}_params"
    current_params = (bool(symbolic), int(seed))
    should_load = (
        current_params != st.session_state.get(params_key)
        or cache_key not in st.session_state
    )
    if should_load:
        with st.spinner("Building schematic…") if not symbolic else nullcontext():
            try:
                st.session_state[cache_key] = load_thesis_diagram_inputs(
                    resolved_config,
                    project_root,
                    seed=int(seed),
                    empirical=not symbolic,
                )
                st.session_state[params_key] = current_params
            except Exception as exc:
                st.session_state.pop(cache_key, None)
                st.session_state.pop(params_key, None)
                st.error(str(exc))
                return

    inputs = st.session_state.get(cache_key)
    if inputs is None:
        return

    fig = build_thesis_figure(inputs)
    st.pyplot(fig, use_container_width=True)

    pdf_bytes = thesis_figure_to_bytes(fig, "pdf")
    png_bytes = thesis_figure_to_bytes(fig, "png", dpi=300)
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="thesis_schematic.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_dl_pdf",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name="thesis_schematic.png",
            mime="image/png",
            key=f"{key_prefix}_dl_png",
            use_container_width=True,
        )
    plt.close(fig)
