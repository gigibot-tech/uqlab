"""Render one model section in the checkpoint arsenal."""

from __future__ import annotations

import streamlit as st

from uqlab.evaluation.pipeline.checkpoint_arsenal import ConfigCluster, ModelSection


def _chip_help(chip) -> str:
    return "\n".join(chip.tooltip_lines)


def _render_shared_baseline(section: ModelSection) -> None:
    baseline = section.shared_baseline
    if not baseline:
        return
    with st.expander(f"Shared setup ({len(baseline)} params)", expanded=False):
        st.caption(" · ".join(f"{k}={v}" for k, v in baseline.items()))


def _render_chip_row(
    cluster: ConfigCluster,
    *,
    key_prefix: str,
    max_visible: int = 24,
) -> str | None:
    selected: str | None = None
    chips = cluster.chips
    visible = chips[:max_visible]
    hidden = chips[max_visible:]

    n = len(visible)
    chip_cols = st.columns(min(n, 12) or 1)
    for i, chip in enumerate(visible):
        with chip_cols[i % len(chip_cols)]:
            if st.button(
                chip.display_label,
                key=f"{key_prefix}_chip_{chip.experiment_id}",
                help=_chip_help(chip),
            ):
                selected = chip.experiment_id

    if hidden:
        with st.expander(f"Show {len(hidden)} more checkpoints in this row"):
            rest_cols = st.columns(min(len(hidden), 12) or 1)
            for i, chip in enumerate(hidden):
                with rest_cols[i % len(rest_cols)]:
                    if st.button(
                        chip.display_label,
                        key=f"{key_prefix}_chip_x_{chip.experiment_id}",
                        help=_chip_help(chip),
                    ):
                        selected = chip.experiment_id
    return selected


def _render_clusters(section: ModelSection, *, key_prefix: str) -> str | None:
    selected: str | None = None
    for cluster in section.config_clusters:
        st.markdown(f"**{cluster.row_header()}**")
        picked = _render_chip_row(cluster, key_prefix=key_prefix)
        if picked:
            selected = picked
    return selected


def render_model_section(
    section: ModelSection,
    *,
    key_prefix: str,
    collapse: bool = True,
) -> str | None:
    """
    Render model header and config-cluster rows with checkpoint buttons.

    Returns selected experiment id if user clicked a chip this run.
    """
    if collapse and section.n_checkpoints > 12:
        with st.expander(
            f"{section.model_label} ({section.n_checkpoints} checkpoints)",
            expanded=False,
        ):
            _render_shared_baseline(section)
            return _render_clusters(section, key_prefix=key_prefix)

    st.markdown(f"#### {section.model_label} ({section.n_checkpoints} checkpoints)")
    _render_shared_baseline(section)
    return _render_clusters(section, key_prefix=key_prefix)
