"""Step 3 — Uncertainty / sweep configuration (perspective-first progressive UI)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from uqlab_orchestrator.config import (
    CIFAR10N_NOISE_LABELS,
    FIXED_REGULAR_TRAIN_PER_CLASS,
    LABEL_NOISE_SWEEP,
    TRAINING_CONFIG,
    aligned_under_train_sweep,
    get_sweep_target,
    launch_mirror_preview,
)
from uqlab_orchestrator.uncertainty import step3_sweep_options
from uqlab_orchestrator.uncertainty.registry import SINGLE_SWEEP_TARGET, SWEEP_BOTH_TARGET
from uqlab_orchestrator.run_spec import filter_under_train_values, is_clean_noise
from uqlab.ui_components.workflow.step3_per_class_table import render_per_class_table
from uqlab.ui_components.workflow.step3_four_region import render_four_region_panel
from uqlab_orchestrator.per_class_sweep import generate_per_class_experiments, get_sweep_summary


def _fixed_conditions_summary(uc: Dict[str, Any]) -> str:
    parts: list[str] = []
    if uc.get("epistemic_enabled"):
        parts.append(
            f"epistemic on ({uc.get('under_supported', '?')}, "
            f"{uc.get('under_train_per_class', '?')}/class)"
        )
    else:
        parts.append("epistemic off")
    if uc.get("aleatoric_enabled"):
        rate = uc.get("custom_noise_rate")
        if rate is not None:
            parts.append(f"aleatoric on ({int(float(rate) * 100)}% noise)")
        else:
            parts.append("aleatoric on")
    else:
        parts.append("aleatoric off")
    return ", ".join(parts)


def _target_label(target: str) -> str:
    for value, label in step3_sweep_options():
        if value == target:
            return label
    return target


def render_step3_collapsed(workflow: Dict[str, Any]) -> None:
    """Collapsed summary when Step 3 is complete."""
    uc = workflow["uncertainty_config"]
    if uc.get("partition_mode") == "four_region":
        st.markdown(
            "**✅ Step 3: Uncertainty** — four-region partition "
            "(noisy / sparse / clean / OOD, single run)"
        )
        return

    target = get_sweep_target(workflow)
    mode = uc.get("sweep_mode", "quick")
    preview = launch_mirror_preview(workflow)

    if target == "single":
        st.markdown(
            f"**✅ Step 3: Uncertainty** — {_target_label(target)} "
            f"({_fixed_conditions_summary(uc)})"
        )
    elif target == SWEEP_BOTH_TARGET:
        primary = preview["primary"]
        st.markdown(
            f"**✅ Step 3: Uncertainty** — {_target_label(target)} "
            f"({mode}, {primary['n_runs']} runs total)"
        )
    else:
        primary = preview["primary"]
        st.markdown(
            f"**✅ Step 3: Uncertainty** — {_target_label(target)} "
            f"({mode}, {primary['n_runs']} pts); "
            f"mirrors: {len(preview.get('mirror_arms') or [])} other type(s)"
        )


def _render_epistemic_panel(
    workflow: Dict[str, Any],
    uc: Dict[str, Any],
    *,
    swept_axis: bool,
) -> tuple[bool, Optional[str], Optional[int], int]:
    """Epistemic controls; under_train_per_class omitted when that axis is swept."""
    st.markdown("##### Epistemic (Fig. 3)")
    if swept_axis:
        st.caption("**Swept** via grid below — set class layout and regular-class pool.")
    else:
        st.caption("**Fixed** mirror — held constant when sweeping label noise (Fig. 4).")

    epistemic_enabled = st.checkbox(
        "Enable epistemic uncertainty (under-trained classes)",
        value=uc.get("epistemic_enabled", False),
    )

    under_supported = None
    under_train_per_class = None
    regular_train_per_class = int(uc.get("regular_train_per_class") or 300)

    if epistemic_enabled:
        under_supported_mode = st.radio(
            "Under-supported classes",
            ["Random selection", "Manual selection"],
            key="step3_epi_mode",
        )
        if under_supported_mode == "Random selection":
            num_under = st.slider("Number of under-supported classes", 1, 5, 2)
            under_supported = f"random:{num_under}"
        else:
            class_names = [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck",
            ]
            default_classes = class_names[:2]
            if uc.get("under_supported") and not str(uc["under_supported"]).startswith("random:"):
                try:
                    default_classes = [
                        class_names[int(x)]
                        for x in str(uc["under_supported"]).split(",")
                        if int(x) < len(class_names)
                    ]
                except ValueError:
                    pass
            selected = st.multiselect(
                "Select under-supported classes",
                class_names,
                default=default_classes,
            )
            under_supported = ",".join(str(class_names.index(c)) for c in selected)

        if not swept_axis:
            under_train_per_class = st.number_input(
                "Samples per under-supported class",
                min_value=10,
                max_value=500,
                value=int(uc.get("under_train_per_class") or 50),
                step=10,
            )
            # Show warning when this is fixed (not swept)
            st.info(
                f"ℹ️ This value ({under_train_per_class}/class) will be **FIXED** across all runs "
                f"when sweeping label noise (Fig. 4)."
            )
        else:
            under_train_per_class = int(uc.get("under_train_per_class") or 50)

        regular_train_per_class = st.number_input(
            "Samples per regular class",
            min_value=50,
            max_value=1000,
            value=int(uc.get("regular_train_per_class") or 300),
            step=50,
        )

    return epistemic_enabled, under_supported, under_train_per_class, regular_train_per_class


def _render_aleatoric_panel(
    workflow: Dict[str, Any],
    uc: Dict[str, Any],
    *,
    swept_axis: bool,
) -> tuple[bool, Optional[float]]:
    """Aleatoric controls; noise % omitted when label noise is the swept axis."""
    st.markdown("##### Aleatoric (Fig. 4)")
    if swept_axis:
        st.caption("**Swept** via grid below — no fixed noise % needed.")
        return True, None

    st.caption("**Fixed** mirror — held constant when sweeping under-train (Fig. 3).")

    ds = workflow.get("dataset_config") or {}
    noise_type = ds.get("noise_type", "clean_label")
    dataset_name = ds.get("dataset_name", "cifar10")
    uses_cifar10n = dataset_name == "cifar10n" and not is_clean_noise(noise_type)

    if uses_cifar10n:
        aleatoric_enabled = st.checkbox(
            f"Use CIFAR-10N noise ({CIFAR10N_NOISE_LABELS.get(noise_type, noise_type)})",
            value=bool(uc.get("aleatoric_enabled", True)),
        )
        custom_noise = None
    elif not is_clean_noise(noise_type):
        aleatoric_enabled = st.checkbox(
            f"Use dataset noise ({noise_type})",
            value=bool(uc.get("aleatoric_enabled", True)),
        )
        custom_noise = None
    else:
        aleatoric_enabled = st.checkbox(
            "Add custom label noise (fixed %)",
            value=bool(uc.get("aleatoric_enabled", False)),
        )
        default_pct = int(float(uc.get("custom_noise_rate") or 0.1) * 100)
        custom_noise = (
            st.slider("Custom noise rate (%)", 0, 50, default_pct, 5) / 100.0
            if aleatoric_enabled
            else None
        )
        # Show warning when noise is fixed (not swept)
        if aleatoric_enabled and custom_noise is not None and not swept_axis:
            st.info(
                f"ℹ️ This noise rate ({int(custom_noise * 100)}%) will be **FIXED** across all runs "
                f"when sweeping under-train (Fig. 3)."
            )

def _render_per_class_mode(workflow: Dict[str, Any]) -> None:
    """Render per-class configuration mode."""
    st.info(
        "💡 **Per-Class Mode**: Configure each class individually. "
        "Enable sweep checkboxes to vary training samples or noise percentage per class."
    )
    
    # Render per-class table (only takes session_key parameter)
    per_class_config, config_changed = render_per_class_table(
        session_key="step3_per_class_config"
    )
    
    if not per_class_config:
        st.warning("⚠️ Per-class configuration is empty. Please configure at least one class.")
        return
    
    # Check if any sweeps are enabled
    has_epistemic_sweep = any(cfg.sweep_epistemic for cfg in per_class_config.values())
    has_aleatoric_sweep = any(cfg.sweep_aleatoric for cfg in per_class_config.values())
    
    if not has_epistemic_sweep and not has_aleatoric_sweep:
        st.info("ℹ️ No sweeps enabled. This will create a single experiment with the configured values.")
    
    # Sweep preset selection (only if sweeps enabled)
    epistemic_preset = "paper"
    aleatoric_preset = "paper"
    
    if has_epistemic_sweep or has_aleatoric_sweep:
        st.markdown("---")
        st.markdown("### 🎚️ Sweep Presets")
        
        col1, col2 = st.columns(2)
        
        if has_epistemic_sweep:
            with col1:
                st.markdown("**Epistemic Sweep (Training Samples)**")
                epistemic_preset = st.radio(
                    "Preset",
                    ["quick", "full", "paper"],
                    index=2,  # default to paper
                    key="step3_pc_epistemic_preset",
                    help="Quick: 3 points, Full: 7 points, Paper: 6 points"
                )
        
        if has_aleatoric_sweep:
            with col2:
                st.markdown("**Aleatoric Sweep (Label Noise %)**")
                aleatoric_preset = st.radio(
                    "Preset",
                    ["quick", "full", "paper"],
                    index=2,  # default to paper
                    key="step3_pc_aleatoric_preset",
                    help="Quick: 3 points, Full: 11 points, Paper: 5 points"
                )
    
    # Generate sweep summary
    try:
        summary = get_sweep_summary(
            per_class_config,
            epistemic_preset=epistemic_preset,
            aleatoric_preset=aleatoric_preset,
        )
        
        st.markdown("---")
        st.markdown("### 📊 Experiment Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Experiments", summary["total_experiments"])
        with col2:
            epistemic_count = len(summary.get("epistemic_classes", []))
            st.metric("Epistemic Sweeps", epistemic_count)
        with col3:
            aleatoric_count = len(summary.get("aleatoric_classes", []))
            st.metric("Aleatoric Sweeps", aleatoric_count)
        
        if summary["total_experiments"] > 0:
            with st.expander("📋 Experiment Breakdown"):
                st.caption(f"Epistemic classes: {summary.get('epistemic_classes', [])}")
                st.caption(f"Aleatoric classes: {summary.get('aleatoric_classes', [])}")
    
    except Exception as e:
        st.error(f"Error generating sweep summary: {e}")
        return
    
    # Continue button
    if st.button("Continue to Evaluation Setup", type="primary", use_container_width=True):
        # Store per-class config in workflow
        workflow["use_per_class_mode"] = True
        workflow["per_class_config"] = per_class_config
        workflow["per_class_epistemic_preset"] = epistemic_preset if has_epistemic_sweep else None
        workflow["per_class_aleatoric_preset"] = aleatoric_preset if has_aleatoric_sweep else None
        workflow["step3_complete"] = True
        
        # Set uncertainty_config for compatibility
        workflow["uncertainty_config"] = {
            "per_class_mode": True,
            "epistemic_enabled": has_epistemic_sweep,
            "aleatoric_enabled": has_aleatoric_sweep,
            "sweep_enabled": has_epistemic_sweep or has_aleatoric_sweep,
        }
        
        st.rerun()


def render_step3_uncertainty(workflow: Dict[str, Any]) -> None:
    """Active Step 3 form — writes ``workflow['uncertainty_config']`` on continue."""
    st.markdown("### Step 3: Uncertainty configuration")

    uc = workflow.get("uncertainty_config") or {}
    partition_mode = st.radio(
        "Partition mode",
        ["legacy", "four_region"],
        index=0 if uc.get("partition_mode", "legacy") != "four_region" else 1,
        format_func=lambda m: (
            "Legacy (Fig. 3/4 sweeps — epistemic + aleatoric)"
            if m == "legacy"
            else "Four-region (noisy / sparse / clean / OOD — single run)"
        ),
        key="step3_partition_mode",
        horizontal=True,
    )

    if partition_mode == "four_region":
        patch = render_four_region_panel(workflow)
        if patch is not None:
            workflow["step3_complete"] = True
            workflow["use_per_class_mode"] = False
            workflow["uncertainty_config"] = {
                **uc,
                **patch,
            }
            st.rerun()
        return

    # Per-Class Mode Toggle
    use_per_class = st.checkbox(
        "🎯 Per-Class Mode (advanced)",
        value=workflow.get("use_per_class_mode", False),
        help="Configure training samples and noise percentage individually for each class"
    )
    
    if use_per_class:
        _render_per_class_mode(workflow)
        return
    
    # Legacy mode (original UI)
    st.info(
        "Pick **what to sweep** (one perspective). The other uncertainty type is your "
        "**fixed mirror** for launch. **Run both** in Step 5 adds the complementary sweep."
    )

    uc = workflow["uncertainty_config"]
    options = step3_sweep_options()
    target_values = [o[0] for o in options]
    current_target = get_sweep_target(workflow)
    if current_target not in target_values:
        current_target = target_values[0]

    target = st.radio(
        "What do you want to sweep?",
        target_values,
        index=target_values.index(current_target),
        format_func=lambda t: _target_label(t),
        help="Each option sweeps one perspective; other types are fixed mirrors.",
    )

    sweep_enabled = target not in (SINGLE_SWEEP_TARGET, SWEEP_BOTH_TARGET) or target == SWEEP_BOTH_TARGET
    if target == SWEEP_BOTH_TARGET:
        sweep_enabled = True
    elif target == "single":
        sweep_enabled = False
    sweep_kind = "label_noise" if target == "label_noise" else "dataset_size"
    if target == "single":
        sweep_kind = uc.get("sweep_kind", "label_noise")
    if target == SWEEP_BOTH_TARGET:
        sweep_kind = "both"

    sweep_mode = st.radio(
        "Sweep grid density",
        ["quick", "full"],
        index=0 if uc.get("sweep_mode", "quick") == "quick" else 1,
        horizontal=True,
        disabled=not sweep_enabled,
        help=(
            f"Quick: noise {LABEL_NOISE_SWEEP['quick']}; "
            f"under-train {aligned_under_train_sweep('quick')}"
        ),
    )

    if sweep_enabled:
        if target == SWEEP_BOTH_TARGET:
            noise_grid = LABEL_NOISE_SWEEP[sweep_mode]
            under_grid = aligned_under_train_sweep(sweep_mode)
            st.caption(
                f"Swept axes: **{len(under_grid)}** under-train → {under_grid} · "
                f"**{len(noise_grid)}** label-noise → {noise_grid}"
            )
        elif target == "label_noise":
            grid = LABEL_NOISE_SWEEP[sweep_mode]
            st.caption(f"Swept axis: **{len(grid)}** label-noise levels → {grid}")
        else:
            vals = aligned_under_train_sweep(sweep_mode)
            st.caption(f"Swept axis: **{len(vals)}** under-train sizes → {vals}")

    # Inline preview (draft workflow for mirror text)
    draft = {**workflow, "uncertainty_config": {**uc, "sweep_target": target, "sweep_enabled": sweep_enabled, "sweep_kind": sweep_kind, "sweep_mode": sweep_mode}}
    preview = launch_mirror_preview(draft)
    st.markdown(preview["inline_primary"])
    if sweep_enabled or target == "single":
        st.caption(preview["inline_mirror"])

    st.markdown("---")

    if target == "single":
        col_epi, col_alea = st.columns(2)
        with col_epi:
            epistemic_enabled, under_supported, under_train_per_class, regular_train_per_class = (
                _render_epistemic_panel(workflow, uc, swept_axis=False)
            )
        with col_alea:
            aleatoric_enabled, custom_noise = _render_aleatoric_panel(
                workflow, uc, swept_axis=False
            )
    elif target == "label_noise":
        epistemic_enabled, under_supported, under_train_per_class, regular_train_per_class = (
            _render_epistemic_panel(workflow, uc, swept_axis=False)
        )
        aleatoric_enabled = True
        custom_noise = None
    elif target == SWEEP_BOTH_TARGET:
        epistemic_enabled, under_supported, under_train_per_class, regular_train_per_class = (
            _render_epistemic_panel(workflow, uc, swept_axis=True)
        )
        aleatoric_enabled = True
        custom_noise = None
    else:  # under_train
        epistemic_enabled, under_supported, under_train_per_class, regular_train_per_class = (
            _render_epistemic_panel(workflow, uc, swept_axis=True)
        )
        aleatoric_enabled, custom_noise = _render_aleatoric_panel(
            workflow, uc, swept_axis=False
        )

    if epistemic_enabled and under_supported and under_train_per_class and regular_train_per_class:
        st.markdown("##### Dataset preview (epistemic)")
        num_under = (
            int(under_supported.split(":")[1])
            if str(under_supported).startswith("random:")
            else len(str(under_supported).split(","))
        )
        num_regular = 10 - num_under
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Under-supported", f"{num_under * under_train_per_class:,} samples")
        with c2:
            st.metric("Regular classes", f"{num_regular * regular_train_per_class:,} samples")
        with c3:
            total = num_under * under_train_per_class + num_regular * regular_train_per_class
            st.metric("Total training", f"{total:,} samples")

    if st.button("Continue to Evaluation Setup", type="primary", use_container_width=True):
        regular = (
            int(regular_train_per_class)
            if epistemic_enabled and regular_train_per_class is not None
            else FIXED_REGULAR_TRAIN_PER_CLASS
        )
        if epistemic_enabled and under_train_per_class is not None:
            if int(under_train_per_class) > regular:
                st.error(
                    f"**under_train_per_class** ({under_train_per_class}) cannot exceed "
                    f"**regular_train_per_class** ({regular}). Lower the epistemic fixed point "
                    "or raise the regular-class budget."
                )
                return

        ep_values = filter_under_train_values(
            aligned_under_train_sweep(sweep_mode),
            regular,
        )
        if target in ("under_train", SWEEP_BOTH_TARGET) and not ep_values:
            st.error(
                f"No under-train sweep values ≤ regular_train_per_class ({regular}). "
                "Adjust Step 3 epistemic settings."
            )
            return

        workflow["step3_complete"] = True
        workflow["uncertainty_config"] = {
            "partition_mode": "legacy",
            "sweep_target": target,
            "epistemic_enabled": epistemic_enabled,
            "under_supported": under_supported if epistemic_enabled else None,
            "under_train_per_class": under_train_per_class if epistemic_enabled else None,
            "regular_train_per_class": (
                regular_train_per_class if epistemic_enabled else regular_train_per_class
            ),
            "aleatoric_enabled": aleatoric_enabled,
            "custom_noise_rate": custom_noise if aleatoric_enabled else None,
            "sweep_enabled": sweep_enabled,
            "sweep_kind": sweep_kind,
            "sweep_mode": sweep_mode,
            "epistemic_sweep_enabled": target in ("under_train", SWEEP_BOTH_TARGET),
            "epistemic_sweep_values": ep_values,
            "aleatoric_sweep_enabled": target in ("label_noise", SWEEP_BOTH_TARGET),
            "aleatoric_sweep_values": LABEL_NOISE_SWEEP[sweep_mode],
        }
        if sweep_enabled:
            preset = TRAINING_CONFIG[sweep_mode]
            workflow["training_config"]["epochs"] = preset["epochs"]
            workflow.setdefault("evaluation_config", {})["mc_passes"] = preset["mc_passes"]
            if target in ("label_noise", SWEEP_BOTH_TARGET):
                workflow["dataset_config"]["noise_type"] = "clean_label"
        st.rerun()
