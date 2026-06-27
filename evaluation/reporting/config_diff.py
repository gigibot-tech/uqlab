"""
Shared config comparison utilities for campaign PDF export and checkpoint arsenal.

Tracked YAML keys, flattening, intersection shared config (campaign), and
mode/median common params (arsenal).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any

from uqlab.experiment_config_flat import normalize_noise_percent
from uqlab.experiment_config_flat import find_config_differences, flatten_dict

TRACKED_PREFIXES = (
    "data.",
    "model.",
    "training.",
    "evaluation.",
)
IGNORE_KEYS = frozenset({
    "paths",
    "paths.data_root",
    "paths.output_dir",
    "paths.checkpoint_dir",
    "seed",
})

PARAM_LABELS: dict[str, str] = {
    "data.dataset_name": "Dataset",
    "data.aleatoric_noise_percentage": "Label noise (%)",
    "data.under_train_per_class": "Under-train / class",
    "data.regular_train_per_class": "Regular-train / class",
    "data.under_supported_classes": "Under-supported classes",
    "data.eval_per_group": "Eval / group",
    "model.architecture": "Architecture",
    "model.dropout": "Dropout",
    "model.hidden_dim": "Hidden dim",
    "training.learning_rate": "Learning rate",
    "training.epochs": "Epochs",
    "training.weight_decay": "Weight decay",
    "training.train_batch_size": "Batch size",
    "evaluation.mc_passes": "MC passes",
    "evaluation.top_k": "Top-k",
    "evaluation.eval_per_group": "Eval / group",
    "model.training_scope": "Training scope",
    "data.noise_type": "Noise type",
}

# Keys shown in arsenal row headers and chip hovers (no nested signal dicts).
ARSENAL_CHIP_KEY_PATHS = frozenset({
    "data.dataset_name",
    "data.noise_type",
    "data.aleatoric_noise_percentage",
    "data.under_train_per_class",
    "data.regular_train_per_class",
    "data.under_supported_classes",
    "data.eval_per_group",
    "model.architecture",
    "model.dropout",
    "model.hidden_dim",
    "model.training_scope",
    "training.epochs",
    "training.learning_rate",
    "training.weight_decay",
    "training.train_batch_size",
    "evaluation.mc_passes",
    "evaluation.top_k",
    "evaluation.attribution_method",
    "evaluation.attribution_backends",
})

ARSENAL_IGNORE_KEY_PREFIXES = (
    "evaluation.signals",
)

SWEEP_KEY_ORDER = (
    "data.aleatoric_noise_percentage",
    "data.under_train_per_class",
    "evaluation.mc_passes",
    "evaluation.top_k",
)


def label_key(key: str) -> str:
    return PARAM_LABELS.get(key, key.replace("_", " ").replace(".", " · "))


def _skip_arsenal_key(key: str) -> bool:
    if key in IGNORE_KEYS or key.startswith("paths."):
        return True
    return any(key == p or key.startswith(f"{p}.") for p in ARSENAL_IGNORE_KEY_PREFIXES)


def tracked_flat(config: dict[str, Any]) -> dict[str, Any]:
    flat = flatten_dict(config)
    out: dict[str, Any] = {}
    for key, value in flat.items():
        if key in IGNORE_KEYS or key.startswith("paths."):
            continue
        if not any(key.startswith(p) or key == p.rstrip(".") for p in TRACKED_PREFIXES):
            continue
        if key == "data.aleatoric_noise_percentage" and value is not None:
            value = normalize_noise_percent(value)
        out[key] = value
    return out


def arsenal_tracked_flat(config: dict[str, Any]) -> dict[str, Any]:
    """Tracked flat for arsenal: skips signal nested dicts and non-surface keys."""
    flat = tracked_flat(config)
    return {
        k: v
        for k, v in flat.items()
        if not _skip_arsenal_key(k) and (k in ARSENAL_CHIP_KEY_PATHS or k.startswith("data."))
    }


def format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return "{" + ", ".join(str(v) for v in value) + "}"
    if isinstance(value, dict):
        return str(value)
    return str(value)


def value_key(value: Any) -> str:
    """Hashable equality key for config values (lists/dicts are not set-safe)."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "[" + ",".join(value_key(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{value_key(v)}" for k, v in sorted(value.items())) + "}"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


@dataclass(frozen=True)
class ConfigChange:
    key: str
    label: str
    old: Any
    new: Any

    def line(self) -> str:
        return f"{self.label}: {format_value(self.old)} → {format_value(self.new)}"


def shared_config(all_flat: list[dict[str, Any]]) -> dict[str, str]:
    """Keys identical across all runs (campaign PDF intersection baseline)."""
    if not all_flat:
        return {}
    keys = set(all_flat[0])
    for flat in all_flat[1:]:
        keys &= set(flat.keys())
    shared: dict[str, str] = {}
    for key in sorted(keys):
        distinct = {value_key(flat.get(key)) for flat in all_flat}
        if len(distinct) == 1:
            shared[label_key(key)] = format_value(all_flat[0].get(key))
    return shared


def _mode_value(values: list[Any]) -> Any:
    if not values:
        return None
    keyed = [(value_key(v), v) for v in values]
    counts = Counter(k for k, _ in keyed)
    best_key = counts.most_common(1)[0][0]
    for k, v in keyed:
        if k == best_key:
            return v
    return values[0]


def _median_value(values: list[Any]) -> Any:
    nums = [float(v) for v in values if v is not None and isinstance(v, (int, float))]
    if nums:
        return median(nums)
    return _mode_value(values)


def common_tracked_params(configs: list[dict[str, Any]]) -> dict[str, str]:
    """
    Mode for categorical / median for numeric — across all configs in a group.

    Returns human-readable labels → formatted values (arsenal model-section baseline).
    """
    if not configs:
        return {}
    flats = [tracked_flat(c) for c in configs]
    all_keys: set[str] = set()
    for flat in flats:
        all_keys |= set(flat.keys())
    out: dict[str, str] = {}
    for key in sorted(all_keys):
        values = [flat.get(key) for flat in flats if key in flat]
        if not values:
            continue
        sample = values[0]
        if isinstance(sample, (int, float)) and not isinstance(sample, bool):
            chosen = _median_value(values)
        else:
            chosen = _mode_value(values)
        out[label_key(key)] = format_value(chosen)
    return out


def _common_raw_by_key(configs: list[dict[str, Any]]) -> dict[str, Any]:
    """Raw keyed common values (dot paths) for diffing."""
    if not configs:
        return {}
    flats = [tracked_flat(c) for c in configs]
    all_keys: set[str] = set()
    for flat in flats:
        all_keys |= set(flat.keys())
    out: dict[str, Any] = {}
    for key in all_keys:
        values = [flat.get(key) for flat in flats if key in flat]
        if not values:
            continue
        sample = values[0]
        if isinstance(sample, (int, float)) and not isinstance(sample, bool):
            out[key] = _median_value(values)
        else:
            out[key] = _mode_value(values)
    return out


def shared_tracked_flat(all_flat: list[dict[str, Any]]) -> dict[str, Any]:
    """Tracked keys with identical values across all flats (raw values)."""
    if not all_flat:
        return {}
    keys = set(all_flat[0])
    for flat in all_flat[1:]:
        keys &= set(flat.keys())
    shared: dict[str, Any] = {}
    for key in sorted(keys):
        distinct = {value_key(flat.get(key)) for flat in all_flat}
        if len(distinct) == 1:
            shared[key] = all_flat[0].get(key)
    return shared


def varying_tracked_labels(all_flat: list[dict[str, Any]]) -> tuple[str, ...]:
    """Human labels for tracked keys that differ within *all_flat*."""
    return tuple(label_key(k) for k in varying_tracked_key_paths(all_flat))


def varying_tracked_key_paths(all_flat: list[dict[str, Any]]) -> tuple[str, ...]:
    """Dot-path keys that differ within *all_flat*."""
    if len(all_flat) < 2:
        return ()
    keys = set(all_flat[0])
    for flat in all_flat[1:]:
        keys |= set(flat.keys())
    varying: list[str] = []
    for key in sorted(keys):
        if not _arsenal_chip_key(key):
            continue
        vals = [flat.get(key) for flat in all_flat if key in flat]
        if len({value_key(v) for v in vals}) > 1:
            varying.append(key)
    return tuple(varying)


def _arsenal_chip_key(key: str) -> bool:
    if _skip_arsenal_key(key):
        return False
    return key in ARSENAL_CHIP_KEY_PATHS


def arsenal_shared_display(all_flat: list[dict[str, Any]]) -> dict[str, str]:
    """Shared params for arsenal row header (surface keys only)."""
    shared = shared_config(all_flat)
    allowed = {label_key(k) for k in ARSENAL_CHIP_KEY_PATHS}
    return {k: v for k, v in shared.items() if k in allowed}


def sweep_hints_from_name(name: str) -> list[str]:
    """Parse sweep value from run name (e.g. fast_alea_*_noise_50)."""
    import re

    hints: list[str] = []
    m = re.search(r"noise_(\d+)", name, re.I)
    if m:
        hints.append(f"Label noise (%): {m.group(1)}")
    m = re.search(r"under_(\d+)", name, re.I)
    if m:
        hints.append(f"Under-train / class: {m.group(1)}")
    return hints


def chip_tooltip_lines(
    *,
    name: str,
    flat: dict[str, Any],
    cluster_flats: list[dict[str, Any]],
    experiment_id: str,
    metric_value: float | None = None,
) -> tuple[str, ...]:
    """
    Human-readable chip hover: this checkpoint's sweep values (not old→new diffs).
    """
    lines: list[str] = []
    title = (name or "").strip()
    if title:
        lines.append(title)

    varying = set(varying_tracked_key_paths(cluster_flats))
    if varying:
        for key in SWEEP_KEY_ORDER:
            if key not in varying:
                continue
            val = flat.get(key)
            if val is not None:
                lines.append(f"{label_key(key)}: {format_value(val)}")
        for key in sorted(varying):
            if key in SWEEP_KEY_ORDER:
                continue
            val = flat.get(key)
            if val is not None:
                lines.append(f"{label_key(key)}: {format_value(val)}")
    else:
        for key in SWEEP_KEY_ORDER:
            val = flat.get(key)
            if val is not None:
                lines.append(f"{label_key(key)}: {format_value(val)}")

    if len(lines) <= (1 if title else 0):
        for hint in sweep_hints_from_name(name):
            if hint not in lines:
                lines.append(hint)

    if metric_value is not None and metric_value == metric_value:
        lines.append(f"Mean AUROC: {metric_value:.3f}")

    short_id = str(experiment_id).replace("-", "")[-8:]
    lines.append(f"ID …{short_id}")
    return tuple(lines)


# Keys that may differ between chips in the same row (typical 1D sweep axes).
CLUSTER_VARIANT_KEYS = frozenset({
    "data.aleatoric_noise_percentage",
    "data.under_train_per_class",
    "evaluation.mc_passes",
    "evaluation.top_k",
})


def chip_display_label(
    flat: dict[str, Any],
    varying_key_paths: tuple[str, ...],
    *,
    short_id: str = "",
) -> str:
    """Human-readable chip button label from sweep axes that vary in the row."""
    varying = set(varying_key_paths)
    parts: list[str] = []
    for key in SWEEP_KEY_ORDER:
        if key not in varying:
            continue
        val = flat.get(key)
        if val is None:
            continue
        if key == "data.aleatoric_noise_percentage":
            parts.append(f"{format_value(val)}%")
        elif key == "data.under_train_per_class":
            parts.append(f"u{format_value(val)}")
        elif key == "evaluation.mc_passes":
            parts.append(f"MC{format_value(val)}")
        elif key == "evaluation.top_k":
            parts.append(f"k{format_value(val)}")
        else:
            parts.append(f"{label_key(key)}={format_value(val)}")

    if parts:
        return "·".join(parts)

    if flat.get("data.aleatoric_noise_percentage") is not None:
        return f"{format_value(flat['data.aleatoric_noise_percentage'])}%"
    if flat.get("data.under_train_per_class") is not None:
        return f"u{format_value(flat['data.under_train_per_class'])}"

    return f"[{short_id}]" if short_id else "checkpoint"


def stable_cluster_fingerprint(flat: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Identity for clustering: all tracked keys except typical sweep axes."""
    return tuple(
        sorted((k, value_key(flat[k])) for k in flat if k not in CLUSTER_VARIANT_KEYS)
    )


def chip_sort_key(flat: dict[str, Any]) -> tuple[float, float, float]:
    """Sort chips by sweep axes (noise, under-train, mc)."""
    noise = flat.get("data.aleatoric_noise_percentage")
    under = flat.get("data.under_train_per_class")
    mc = flat.get("evaluation.mc_passes")
    return (
        float(noise) if noise is not None else -1.0,
        float(under) if under is not None else -1.0,
        float(mc) if mc is not None else -1.0,
    )


def diffs_from_common(
    configs: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[ConfigChange, ...]:
    """Params in *config* that differ from the mode/median common baseline."""
    common_raw = _common_raw_by_key(configs)
    flat = tracked_flat(config)
    changes: list[ConfigChange] = []
    for key in sorted(set(common_raw) | set(flat)):
        old = common_raw.get(key)
        new = flat.get(key)
        if value_key(old) == value_key(new):
            continue
        changes.append(ConfigChange(key, label_key(key), old, new))
    return tuple(changes)


def find_tracked_differences(config1: dict[str, Any], config2: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Tracked-key differences between two full configs."""
    changes: list[tuple[str, Any, Any]] = []
    for key, old, new in find_config_differences(config1, config2):
        if key in IGNORE_KEYS or key.startswith("paths."):
            continue
        if not any(key.startswith(p) for p in TRACKED_PREFIXES):
            continue
        if key == "data.aleatoric_noise_percentage":
            old = normalize_noise_percent(old)
            new = normalize_noise_percent(new)
        changes.append((key, old, new))
    return changes


def disambiguate_suffix_ids(ids: list[str], *, start: int = 4, max_len: int = 12) -> dict[str, str]:
    """Map full experiment id → shortest unique suffix within the group."""
    raw_map = {i: str(i).replace("-", "") for i in ids}
    for n in range(start, max_len + 1):
        labels = {full: raw[-n:] for full, raw in raw_map.items()}
        if len(set(labels.values())) == len(labels):
            return labels
    return {full: raw_map[full][-max_len:] for full in ids}
