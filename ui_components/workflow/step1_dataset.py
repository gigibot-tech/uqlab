"""Step 1 — Dataset selection (progressive workflow UI)."""

from __future__ import annotations

from typing import Any, Callable, Dict

import streamlit as st

from uqlab_orchestrator.config import (
    CIFAR10N_NOISE_LABELS,
    DATASET_CATALOG,
)
from uqlab_orchestrator.dataset_facade import get_dataset_spec, is_clean_noise


def render_step1_dataset(
    workflow: Dict[str, Any],
    *,
    fetch_stats: Callable[[str, str], dict],
) -> bool:
    """
    Render Step 1 dataset picker. Returns True if the flow should stop here.
    """
    st.markdown("### 📊 Step 1: Dataset Selection")

    dataset_ids = list(DATASET_CATALOG.keys())
    saved_ds = workflow["dataset_config"].get("dataset_name", "cifar10")
    ds_index = dataset_ids.index(saved_ds) if saved_ds in dataset_ids else 0
    dataset_choice = st.selectbox(
        "Choose a dataset",
        dataset_ids,
        index=ds_index,
        format_func=lambda k: DATASET_CATALOG[k]["label"],
        help="Registry-driven list — MNIST uses synthetic noise only.",
    )
    st.caption(DATASET_CATALOG[dataset_choice]["description"])
    spec = get_dataset_spec(dataset_choice)

    if spec.supports_human_noise:
        noise_options = list(spec.noise_options)
        saved_noise = workflow["dataset_config"].get("noise_type", "worse_label")
        if saved_noise == "none":
            saved_noise = "worse_label"
        default_noise_idx = (
            noise_options.index(saved_noise)
            if saved_noise in noise_options
            else noise_options.index("worse_label")
        )
        noise_choice = st.selectbox(
            "CIFAR-10N noise split",
            noise_options,
            index=default_noise_idx,
            format_func=lambda k: CIFAR10N_NOISE_LABELS.get(k, k),
            help="Human-annotated label noise from the CIFAR-10N benchmark (Wei et al.).",
        )
        if is_clean_noise(noise_choice):
            st.caption("Clean split — equivalent to CIFAR-10 labels on the same images.")
        else:
            st.caption(
                "Training/eval will use the selected human-noise split. "
                "For paper-style synthetic sweeps, pick **CIFAR-10 (original)** or **MNIST**."
            )
    else:
        noise_choice = "clean_label"
        if dataset_choice == "cifar10":
            st.info(
                "ℹ️ **CIFAR-10 (original)** uses clean labels only. "
                "Label noise for Fig. 4 sweeps is injected synthetically in Step 3."
            )
        elif dataset_choice == "mnist":
            st.info(
                "ℹ️ **MNIST** uses clean labels; Fig. 4 sweeps inject synthetic uniform label noise."
            )

    with st.spinner("Loading dataset statistics..."):
        stats = fetch_stats(dataset_choice, noise_choice)

    if stats:
        if stats.get("source") in ("fallback", "local"):
            st.caption("Using offline/local stats (backend may not list this dataset yet).")
        st.markdown("#### Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", f"{stats.get('total_samples', 0):,}")
        with col2:
            st.metric("Classes", stats.get("num_classes", spec.num_classes))
        with col3:
            noise_rate = stats.get("noise_rate", 0.0)
            st.metric(
                "Noise Rate",
                f"{noise_rate:.1%}" if not is_clean_noise(noise_choice) else "0%",
            )

        with st.expander("📋 View dataset details"):
            st.json(stats)

        if st.button("✓ Continue to Training Setup", type="primary", use_container_width=True):
            workflow["step1_complete"] = True
            workflow["dataset_config"] = {
                "dataset_name": dataset_choice,
                "noise_type": noise_choice,
                "stats": stats,
            }
            st.rerun()

    return True
