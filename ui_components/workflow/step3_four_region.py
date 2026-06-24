"""Step 3 — four-region partition editor."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from uqlab.data.class_regions import (
    ALL_REGIONS,
    DEFAULT_FOUR_REGION_PRESET,
    validate_class_regions,
)

_CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _class_names_for_workflow(workflow: Dict[str, Any]) -> List[str]:
    ds = workflow.get("dataset_config") or {}
    num_classes = int((ds.get("stats") or {}).get("num_classes") or 10)
    dataset_name = str(ds.get("dataset_name") or "cifar10").lower()
    if dataset_name == "fashion_mnist":
        base = list(_FASHION_MNIST_CLASS_NAMES)
    else:
        base = list(_CIFAR10_CLASS_NAMES)
    if len(base) < num_classes:
        base.extend(str(i) for i in range(len(base), num_classes))
    return base[:num_classes]


def _class_options(class_names: List[str]) -> List[str]:
    return [f"{i}: {name}" for i, name in enumerate(class_names)]


def _parse_class_selection(selected: List[str]) -> List[int]:
    return [int(s.split(":")[0]) for s in selected]


def render_four_region_panel(workflow: Dict[str, Any]) -> dict[str, Any] | None:
    """
    Edit four-region partition.

    Returns an ``uncertainty_config`` patch when the user clicks Continue and the
    partition is valid; otherwise ``None``.
    """
    uc = workflow.get("uncertainty_config") or {}
    stored = uc.get("class_regions") or DEFAULT_FOUR_REGION_PRESET
    regular_train = int(uc.get("regular_train_per_class") or 300)
    class_names = _class_names_for_workflow(workflow)
    num_classes = len(class_names)

    st.markdown(f"##### Four-region partition ({num_classes} classes)")
    st.caption(
        f"Assign each class 0–{num_classes - 1} to **noisy**, **sparse**, **clean**, or **OOD**. "
        "All four eval pools are populated in a single run."
    )

    class_regions: dict[str, dict[str, Any]] = {}
    for region in ALL_REGIONS:
        preset = dict(stored.get(region) or DEFAULT_FOUR_REGION_PRESET[region])
        default_ids = [int(c) for c in preset.get("classes") or [] if int(c) < num_classes]
        default_labels = [f"{i}: {class_names[i]}" for i in default_ids]

        with st.expander(region.title(), expanded=(region == "noisy")):
            picked = st.multiselect(
                f"Classes ({region})",
                _class_options(class_names),
                default=default_labels,
                key=f"step3_region_{region}_classes",
            )
            spec: dict[str, Any] = {"classes": _parse_class_selection(picked)}
            if region == "noisy":
                spec["label_flip_pct"] = st.slider(
                    "Label flip % (noisy region)",
                    0,
                    100,
                    int(float(preset.get("label_flip_pct") or 30)),
                    5,
                    key="step3_region_noisy_flip",
                )
                spec["train_fraction"] = 1.0
            elif region == "ood":
                spec["train_fraction"] = 0.0
            else:
                default_tf = float(preset.get("train_fraction") or (0.10 if region == "sparse" else 1.0))
                spec["train_fraction"] = st.slider(
                    f"Train fraction ({region})",
                    0.0,
                    1.0,
                    default_tf,
                    0.05,
                    key=f"step3_region_{region}_train_frac",
                )
            class_regions[region] = spec

    regular_train_per_class = st.number_input(
        "Max training samples per class (noisy/clean full regions)",
        min_value=50,
        max_value=1000,
        value=regular_train,
        step=50,
        key="step3_four_region_regular",
    )

    try:
        validate_class_regions(class_regions, num_classes=num_classes)
    except ValueError as exc:
        st.error(str(exc))
        return None

    if st.button(
        "Continue to Evaluation Setup",
        type="primary",
        use_container_width=True,
        key="step3_four_region_continue",
    ):
        sparse_classes = class_regions["sparse"]["classes"]
        sparse_fraction = float(class_regions["sparse"]["train_fraction"])
        return {
            "partition_mode": "four_region",
            "class_regions": class_regions,
            "sweep_target": "single",
            "sweep_enabled": False,
            "sweep_kind": "four_region",
            "epistemic_enabled": True,
            "under_supported": ",".join(str(c) for c in sparse_classes),
            "under_train_per_class": max(
                1, int(regular_train_per_class * sparse_fraction)
            ),
            "regular_train_per_class": int(regular_train_per_class),
            "aleatoric_enabled": True,
            "custom_noise_rate": None,
            "epistemic_sweep_enabled": False,
            "aleatoric_sweep_enabled": False,
        }
    return None
